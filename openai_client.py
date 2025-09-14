import os
import time
import logging
from collections import deque
from openai import OpenAI
try:
    from openai import RateLimitError, APIStatusError
except Exception:
    RateLimitError = tuple()
    APIStatusError = tuple()

logger = logging.getLogger(__name__)

CORE_PROMPT = os.getenv("OPENAI_CORE_PROMPT", "").strip()
ROLE_PROMPT = os.getenv("OPENAI_ROLE_PROMPT", "").strip()
FALLBACK_GENERIC   = os.getenv("FALLBACK_GENERIC", "処理中に問題が発生しました。恐れ入りますが、もう一度入力していただけますか？").strip()
FALLBACK_RATE      = os.getenv("FALLBACK_RATE", "ただいま少し混み合っているため、すぐに応答できませんでした。時間をおいてから、もう一度お試しください。").strip()
FALLBACK_SENSITIVE = os.getenv("FALLBACK_SENSITIVE", "").strip()

class TokenMeter:
    def __init__(self):
        self.buf = deque()
        self.total = 0

    def _prune(self, now: float):
        # 60秒より古い要素を落として合計を調整
        while self.buf and now - self.buf[0][0] > 60.0:
            t, n = self.buf.popleft()
            self.total -= n
            if self.total < 0:
                self.total = 0

    def add(self, tokens: int):
        now = time.time()
        self._prune(now)
        self.buf.append((now, int(tokens)))
        self.total += int(tokens)

    def last_60s(self) -> int:
        self._prune(time.time())
        return self.total

_token_meter = TokenMeter()

class OpenAIClient:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")

    def _is_rate_limit(self, err: Exception) -> bool:
        try:
            if isinstance(err, RateLimitError):
                return True
        except Exception:
            pass

        try:
            if isinstance(err, APIStatusError) and getattr(err, "status_code", None) == 429:
                return True
        except Exception:
            pass

        msg = (str(err) or "").lower()
        if "429" in msg or "rate limit" in msg or "too many requests" in msg:
            return True
        if hasattr(err, "status_code") and getattr(err, "status_code") == 429:
            return True
        return False

    def _compose_messages(self, user_message: str):
        messages = []
        if CORE_PROMPT:
            messages.append({"role": "system", "content": CORE_PROMPT})
        if ROLE_PROMPT:
            messages.append({"role": "system", "content": ROLE_PROMPT})
        messages.append({"role": "user", "content": user_message})
        return messages

    def _extract_usage(self, response):
        try:
            u = getattr(response, "usage", None) or {}
            inp = getattr(u, "input_tokens", None) or u.get("input_tokens")
            out = getattr(u, "output_tokens", None) or u.get("output_tokens")
            tot = getattr(u, "total_tokens", None) or u.get("total_tokens")

            # total が無ければ加算で推定
            if tot is None and (inp is not None or out is not None):
                tot = (inp or 0) + (out or 0)

            # 何も取れない場合は None 返し
            if inp is None and out is None and tot is None:
                return (None, None, None)

            return (int(inp) if inp is not None else None,
                    int(out) if out is not None else None,
                    int(tot) if tot is not None else None)
        except Exception:
            return (None, None, None)

    def _friendly_fallback(self, err: Exception | None, kind: str = "generic") -> str:
        if err and self._is_rate_limit(err):
            return FALLBACK_RATE
        if kind == "sensitive":
            return FALLBACK_SENSITIVE
        if kind == "rate":
            return FALLBACK_RATE
        msg = (str(err) or "").lower() if err else ""
        if ("policy" in msg or ("content" in msg and "filter" in msg)):
            return FALLBACK_SENSITIVE
        if ("rate" in msg or "429" in msg or "timeout" in msg or "temporarily" in msg or "503" in msg):
            return FALLBACK_RATE
        return FALLBACK_GENERIC

    def get_reply(self, user_message: str, previous_response_id: str=None):
        try:
            messages = self._compose_messages(user_message)

            if previous_response_id:
                response = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    previous_response_id=previous_response_id
                )
            else:
                response = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    store=True
                )

            ai_message  = (getattr(response, "output_text", "") or "").strip()
            response_id = getattr(response, "id", None)

            inp_toks, out_toks, tot_toks = self._extract_usage(response)
            if tot_toks is not None:
                _token_meter.add(tot_toks)

            logger.info(
                "🧮 usage: input=%s output=%s total=%s | last60s=%s",
                inp_toks, out_toks, tot_toks, _token_meter.last_60s()
            )

            if not ai_message:
                logger.warning("⚠️ output_text is empty; using generic fallback")
                return self._friendly_fallback(None, "generic"), (response_id or previous_response_id)

            return ai_message, response_id

        except Exception as e:
            logger.exception("🔥 OpenAIClient#get_reply failed")
            if self._is_rate_limit(e):
                return self._friendly_fallback(e, "rate"), previous_response_id
            return self._friendly_fallback(e), previous_response_id

import os
import time
import math
import logging
import tiktoken
from collections import deque
from typing import List, Dict, Tuple
from openai import OpenAI, BadRequestError
try:
    from openai import RateLimitError, APIStatusError
except Exception:
    RateLimitError = tuple()
    APIStatusError = tuple()

logger = logging.getLogger(__name__)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # 通常応答用
MODEL_SUMMARY = os.getenv("OPENAI_MODEL_SUMMARY", "gpt-4.1-nano")  # 要約専用
MODEL_CONTEXT = 128_000
BUDGET_RATIO = 0.80             # 上限の 80%
OUTPUT_RESERVE = 1024           # 応答用に最低これだけは確保
TARGET_BUDGET = max(16_000, int(MODEL_CONTEXT * BUDGET_RATIO) - OUTPUT_RESERVE)

client = OpenAI()

CORE_PROMPT = os.getenv("OPENAI_CORE_PROMPT", "").strip()
ROLE_PROMPT = os.getenv("OPENAI_ROLE_PROMPT", "").strip()
FALLBACK_GENERIC   = os.getenv("FALLBACK_GENERIC", "処理中に問題が発生しました。恐れ入りますが、もう一度入力していただけますか？").strip()
FALLBACK_RATE      = os.getenv("FALLBACK_RATE", "ただいま少し混み合っているため、すぐに応答できませんでした。時間をおいてから、もう一度お試しください。").strip()
FALLBACK_SENSITIVE = os.getenv("FALLBACK_SENSITIVE", "").strip()


def estimate_tokens(messages: List[Dict]) -> int:
        """
        ざっくり推定。モデルに対応するエンコーダが無ければ cl100k で代替。
        """
        try:
            enc = tiktoken.encoding_for_model(MODEL)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")

        total = 0
        for m in messages:
            content = m.get("content") or ""
            if isinstance(content, list):
                text = " ".join([p.get("text","") for p in content if isinstance(p, dict)])
            else:
                text = str(content)
            total += len(enc.encode(m.get("role","") + ": " + text))
        return total

def _system_block(title: str, body: str) -> Dict:
        return {"role": "system", "content": f"### {title}\n{body}".strip()}

def summarize_text_block(text: str, lang: str = "ja") -> str:
        """
        古い区間を要約（事実/決定/タスク/固有名詞に絞って短文化）
        """
        system = "You are a careful summarizer."
        prompt = (
            "以下の会話ログを、事実・決定事項・未決タスク・固有名詞に絞って"
            "日本語で150〜250トークン程度に要約してください。挨拶や冗長表現は削除。引用は不要。"
        )
        # Chat Completions でも Responses でもOKにする簡易実装
        try:
            resp = client.chat.completions.create(
                model=MODEL_SUMMARY,
                messages=[
                    {"role":"system","content":system},
                    {"role":"user","content": prompt + "\n\n" + text}
                ],
                temperature=0.2,
                max_tokens=400,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            # フェイルセーフでそのまま切り詰め
            return text[:1500]

def shrink_context(msgs: List[Dict], target_tokens: int) -> List[Dict]:
        """
        いちばん古い user/assistant ペアからまとめて要約に置換していく。
        """
        # system 以外の最初の連続ブロックをまとめる
        sys_count = sum(1 for m in msgs if m["role"] == "system")
        # 古い順で非systemを収集
        non_sys = [i for i, m in enumerate(msgs) if m["role"] != "system"]
        if not non_sys:
            return msgs

        start = non_sys[0]  # 一番古い
        # ある程度の塊（例: 4〜6メッセージ）で要約
        end = min(start + 6, len(msgs))
        chunk = msgs[start:end]
        text = "\n\n".join([f"{m['role']}: {m['content']}" for m in chunk])
        summary = summarize_text_block(text)

        # 置換（systemの直後あたりに「Summarized history」ブロックを差し込む）
        new_msgs = msgs[:start] + [_system_block("Summarized history (older turns)", summary)] + msgs[end:]

        # まだ大きければ繰り返す
        while estimate_tokens(new_msgs) > target_tokens:
            # 次の古い塊
            non_sys2 = [i for i, m in enumerate(new_msgs) if m["role"] != "system"]
            if len(non_sys2) <= 6:
                break
            start2 = non_sys2[0]
            end2 = min(start2 + 6, len(new_msgs))
            chunk2 = new_msgs[start2:end2]
            text2 = "\n\n".join([f"{m['role']}: {m['content']}" for m in chunk2])
            summary2 = summarize_text_block(text2)
            new_msgs = new_msgs[:start2] + [_system_block("Summarized history (older turns)", summary2)] + new_msgs[end2:]

            if all(m["role"] == "system" for m in new_msgs):
                break

        return new_msgs

def force_shrink(msgs: List[Dict]) -> List[Dict]:
        """
        最終手段：既存のサマリを再要約 or 非systemをほぼ落とす
        """
        # 既存 summary を短縮
        for i, m in enumerate(msgs):
            if m["role"] == "system" and "summary" in m.get("content","").lower():
                m["content"] = m["content"][:2000]  # 大雑把に切詰め
        # まだダメなら直近以外の user/assistant を削る
        if estimate_tokens(msgs) > TARGET_BUDGET:
            keep = []
            # system 全部 + 直近2 user/assistant
            non_sys = [m for m in msgs if m["role"] != "system"]
            keep_non_sys = non_sys[-4:]
            keep = [m for m in msgs if m["role"] == "system"] + keep_non_sys
            return keep
        return msgs

def build_messages(
        system_prompt: str,
        profile_bullets: List[str],
        running_summary: str,
        recent_turns: List[Tuple[str, str]],
        hard_cap_tokens: int = TARGET_BUDGET,
        ) -> List[Dict]:
        """
        recent_turns: [(role, content)] 古→新の順 でも 新→古でもOK。後で並び替える。
        """
        msgs: List[Dict] = []
        # 1) 最小セット まず CORE / ROLE / SYSTEM をこの順で差し込む（存在するもののみ）
        if CORE_PROMPT:
            msgs.append({"role": "system", "content": CORE_PROMPT[:4000]})
        if ROLE_PROMPT:
            msgs.append({"role": "system", "content": ROLE_PROMPT[:4000]})
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt.strip()[:4000]})


        if profile_bullets:
            profile_text = "- " + "\n- ".join(profile_bullets[:10])
            msgs.append(_system_block("User profile (concise)", profile_text))

        if running_summary:
            msgs.append(_system_block("Conversation background (summary)", running_summary))

        # 2) 直近Kターン（多めに入れて、予算で削る）
        #    新しい順で最後に来るように整列
        turns = recent_turns[:]  # shallow copy
        # （古→新）の想定にして、新しいものが後に並ぶようにしておく
        for role, content in turns:
            msgs.append({"role": role, "content": content})

        # 3) 超過チェック → 縮約
        tokens = estimate_tokens(msgs)
        if tokens > hard_cap_tokens:
            msgs = shrink_context(msgs, hard_cap_tokens)

        # 念押し
        while estimate_tokens(msgs) > hard_cap_tokens and any(m["role"] != "system" for m in msgs):
            msgs = force_shrink(msgs)

        return msgs

def _extract_usage_from_chat_completion(resp):
    try:
        u = getattr(resp, "usage", None)
        if not u:
            return (None, None, None)

        def uget(key):
            try:
                v = getattr(u, key)
                if v is not None:
                    return v
            except Exception:
                pass
            if isinstance(u, dict):
                return u.get(key)
            return None

        inp = uget("prompt_tokens")
        out = uget("completion_tokens")
        tot = uget("total_tokens")

        if inp is None:
            inp = uget("input_tokens")
        if out is None:
            out = uget("output_tokens")
        if tot is None and (inp is not None or out is not None):
            tot = (inp or 0) + (out or 0)

        return (
            int(inp) if inp is not None else None,
            int(out) if out is not None else None,
            int(tot) if tot is not None else None,
        )
    except Exception:
        return (None, None, None)


def generate_chat(messages: List[Dict], max_tokens: int = 1024, temperature: float = 0.2) -> str:
        """
        実際の生成呼び出し。context超過時はフェイルセーフで縮約再送。
        """
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            inp_toks, out_toks, tot_toks = _extract_usage_from_chat_completion(resp)
            if tot_toks is not None:
                _token_meter.add(tot_toks)
            logger.info("🧮 usage: input=%s output=%s total=%s | last60s=%s",
                    inp_toks, out_toks, tot_toks, _token_meter.last_60s())
            return resp.choices[0].message.content
        except BadRequestError as e:
            # context_length_exceeded 対応：最小構成に縮約して再送
            if "context length" in str(e).lower() or "context_length_exceeded" in str(e).lower():
                minimal = force_shrink(messages)
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=minimal,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                inp_toks, out_toks, tot_toks = _extract_usage_from_chat_completion(resp)
                if tot_toks is not None:
                    _token_meter.add(tot_toks)
                logger.info("🧮 usage: input=%s output=%s total=%s | last60s=%s",
                        inp_toks, out_toks, tot_toks, _token_meter.last_60s())
                return resp.choices[0].message.content
            raise

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
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

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

import os
import uvicorn
import requests
import logging
import random
from fastapi import FastAPI, Request
from pydantic import BaseModel
from openai_client import OpenAIClient, build_messages, generate_chat, estimate_tokens, summarize_text_block
from firestore_db import FirestoreDB, get_user_profile, get_running_summary, update_running_summary
from typing import List, Tuple

app = FastAPI()
openai_client = OpenAIClient()
db = FirestoreDB()

LINE_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN")
STICKER_REPLIES = [
    os.getenv("LINE_STICKER_REPLY01", "").strip(),
    os.getenv("LINE_STICKER_REPLY02", "").strip(),
    os.getenv("LINE_STICKER_REPLY03", "").strip(),
]

STICKER_REPLIES = [r for r in STICKER_REPLIES if r]

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

class LineWebhookEvent(BaseModel):
    replyToken: str
    type: str
    message: dict
    source: dict

class LineWebhookBody(BaseModel):
    events: list[LineWebhookEvent]

@app.get("/")
def read_root():
    return {"message": "LINE chatbot is up and running."}

@app.post("/webhook")
async def webhook(body: LineWebhookBody):
    logger.info("📩 /webhook called")
    if not body.events:
        return {"status": "no events"}

    event = body.events[0]

    if event.type == "message":
        mtype = event.message.get("type")

        if mtype == "sticker":
            logger.info(f"🧩 sticker from {event.source['userId']}: "
                        f"pkg={event.message.get('packageId')} sid={event.message.get('stickerId')}")
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"
            }

            reply_text = random.choice(STICKER_REPLIES) if STICKER_REPLIES else "スタンプありがとう☺️"

            payload = {
                "replyToken": event.replyToken,
                "messages": [{"type": "text", "text": reply_text}]
            }

            requests.post("https://api.line.me/v2/bot/message/reply", headers=headers, json=payload)
            return {"status": "success"}

        if mtype == "text":
            user_id = event.source["userId"]
            user_message = event.message["text"]
            logger.info(f"👤 From {user_id}: {user_message}")

            # --- ここから差し替えブロック ---------------------------------------
            # 1) Firestore からプロフィール＆ランニング要約
            profile_bullets = get_user_profile(user_id)  # ["氏名: ...", "会社: ...", ...]
            running_summary, last_idx = get_running_summary(user_id)  # thread_id として user_id を使う運用

            # 2) 直近Kターン（例：30）を取得して今回の user 発話を追加
            #    db 側に get_recent_history(user_id, limit) を用意（無ければ実装してください）
            #    戻りは [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}] 形式を想定
            try:
                recent_history = db.get_recent_history(user_id, limit=30)
            except AttributeError:
                # フォールバック：過去ログを保持していない場合は空配列で進める
                recent_history = []
            recent_turns = [(h["role"], h["content"]) for h in recent_history] + [("user", user_message)]

            # 3) コンテキスト構築（ローリング＋予算＋段階的縮約）
            SYSTEM_PROMPT = "あなたは有能なアシスタントです。丁寧かつ簡潔に日本語で回答します。"
            messages = build_messages(
                system_prompt=SYSTEM_PROMPT,
                profile_bullets=profile_bullets,       # “忘れない事実メモ” を system 直下に差し込み
                running_summary=running_summary,       # 古い経緯は要約として注入
                recent_turns=recent_turns,             # 直近Kターン＋今回の入力
            )

            # 4) 生成（context超過時は generate_chat 内で最小構成にフェイルセーフ）
            ai_response = generate_chat(messages, max_tokens=1024, temperature=0.3)

            # 5) 会話ログ保存（あなたの既存I/Fに合わせて）
            db.log_conversation(user_id, user_message, ai_response, response_id=None)

            # 6) 生履歴が膨らんだら、古い半分をサマリに吸収して Firestore 更新
            raw_tokens = estimate_tokens([{"role": r, "content": c} for r, c in recent_turns])
            if raw_tokens > 60_000:  # 閾値は運用で調整。gpt-4o-mini なら 60k〜80k 目安
                # 例：history の前半をひとまとめにして要約→running_summary に追記
                half = max(1, len(recent_history) // 2)
                if half > 0:
                    old_block = "\n\n".join([f"{h['role']}: {h['content']}" for h in recent_history[:half]])
                    addition = summarize_text_block(old_block)
                    new_summary = (running_summary + "\n\n" + addition).strip() if running_summary else addition
                    update_running_summary(user_id, new_summary, last_idx + half)

            # 7) LINE 返信
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"
            }
            payload = {
                "replyToken": event.replyToken,
                "messages": [{"type": "text", "text": ai_response}]
            }
            requests.post("https://api.line.me/v2/bot/message/reply", headers=headers, json=payload)

            logger.info("✅ webhook text flow completed")
            return {"status": "success"}

    return {"status": "unsupported message type"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 10000)), reload=True)

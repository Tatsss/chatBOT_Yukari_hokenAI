from fastapi import FastAPI, Request
from pydantic import BaseModel
from openai_client import OpenAIClient
from firestore_db import FirestoreDB
import os
import uvicorn
import requests
import logging
import random

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

            last_response_id = db.get_last_response_id(user_id)
            logger.info(f"⭐️ last_response_id = {last_response_id!r}")

            ai_response, response_id = openai_client.get_reply(user_message, last_response_id)
            logger.info(f"🤖 OpenAI replied: response_id={response_id!r}")

            db.log_conversation(user_id, user_message, ai_response, response_id)
            logger.info("💾 log_conversation completed")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"
            }
            payload = {
                "replyToken": event.replyToken,
                "messages": [{"type": "text", "text": ai_response}]
            }
            requests.post("https://api.line.me/v2/bot/message/reply", headers=headers, json=payload)
            return {"status": "success"}

    return {"status": "unsupported message type"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 10000)), reload=True)

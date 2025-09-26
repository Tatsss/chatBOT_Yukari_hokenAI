import os
import json, logging
import firebase_admin
from datetime import datetime
from typing import List, Tuple, Optional
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter

logger = logging.getLogger(__name__)

def _ensure_app():
    if not firebase_admin._apps:
        cred_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
        if not cred_json:
            raise ValueError("GOOGLE_CREDENTIALS_JSON が未設定です。")
        cred_dict = json.loads(cred_json)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)

_ensure_app()
db = firestore.client()

class FirestoreDB:
    def __init__(self):
        self.db = db
        self.collection = self.db.collection("conversations")

    def get_last_response_id(self, user_id):
        docs = (
            self.collection
            .where(filter=FieldFilter("user_id", "==", user_id))
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(1)
            .stream()
        )
        for doc in docs:
            return (doc.to_dict() or {}).get("response_id")
        return None

    def log_conversation(self, user_id, user_message, ai_response, response_id):
        try:
            self.collection.add({
                "user_id": user_id,
                "user_message": user_message,
                "ai_response": ai_response,
                "response_id": response_id,
                "timestamp": firestore.SERVER_TIMESTAMP,
            })

        except Exception:
            logger = logging.getLogger(__name__)
            logger.exception("🔥 FirestoreDB.log_conversation failed")
            raise

    def get_recent_history(self, user_id: str, limit: int = 30):
        """
        log_conversation の構造（1 doc に user/assistant ペア）に合わせて
        直近 N 件から role/content の配列を組み立てる。
        """

        snaps = (
            self.collection
            .where(filter=FieldFilter("user_id", "==", user_id))
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        pairs = []
        for s in snaps:
            d = s.to_dict() or {}
            um = d.get("user_message"); ar = d.get("ai_response")
            if um: pairs.append({"role":"user","content":um})
            if ar: pairs.append({"role":"assistant","content":ar})
        pairs.reverse()  # 古い→新しい
        return pairs

# --- Profile memory ---

def get_user_profile(uid: str) -> List[str]:
        """users/{uid}/profile_memory.documents('facts') の配列を返す。無ければ空配列。"""
        ref = db.collection("users").document(uid).collection("profile_memory").document("facts")
        doc = ref.get()
        if doc.exists:
            data = doc.to_dict() or {}
            return data.get("facts", [])
        return []

def upsert_user_profile(uid: str, facts: List[str]) -> None:
        """箇条書きの事実メモを保存。10行/200-400トークン程度を目安に保つ。"""
        ref = db.collection("users").document(uid).collection("profile_memory").document("facts")
        ref.set({
            "facts": facts[:50],  # 念のため上限
            "updated_at": datetime.utcnow().isoformat()
        }, merge=True)

def get_running_summary(thread_id: str) -> Tuple[str, int]:
        """
        conversations/{threadId}/running_summary:
        text: str（サマリ本文）
        last_index: int（どのmessageIndexまで要約済みか）
        """
        ref = db.collection("conversations").document(thread_id).collection("meta").document("running_summary")
        doc = ref.get()
        if doc.exists:
            data = doc.to_dict() or {}
            return data.get("text", ""), int(data.get("last_index", -1))
        return "", -1

def update_running_summary(thread_id: str, text: str, last_index: int) -> None:
        ref = db.collection("conversations").document(thread_id).collection("meta").document("running_summary")
        ref.set({
            "text": text,
            "last_index": last_index,
            "updated_at": datetime.utcnow().isoformat()
        }, merge=True)

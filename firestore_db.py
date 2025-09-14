import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

class FirestoreDB:
    def __init__(self):
        if not firebase_admin._apps:
            cred_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
            if not cred_json:
                raise ValueError("GOOGLE_CREDENTIALS_JSON 環境変数が設定されていません。")

            cred_dict = json.loads(cred_json)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)

        self.db = firestore.client()
        self.collection = self.db.collection("conversations")

    def get_last_response_id(self, user_id):
        docs = (
            self.collection.where("user_id", "==", user_id)
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(1)
            .stream()
        )
        for doc in docs:
            return doc.to_dict().get("response_id")
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

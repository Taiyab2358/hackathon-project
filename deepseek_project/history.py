# history.py

class ChatHistory:
    def __init__(self):
        self.histories = {}  # { session_id: [ {role, content}, ... ] }

    def add_user_message(self, session_id: str, message: str):
        self.histories.setdefault(session_id, [])
        self.histories[session_id].append({"role": "user", "content": message})

    def add_bot_message(self, session_id: str, message: str):
        self.histories.setdefault(session_id, [])
        self.histories[session_id].append({"role": "bot", "content": message})

    def get_formatted_history(self, session_id: str) -> str:
        """
        Returns conversation as a single string prompt.
        Useful for passing context to the model.
        """
        history = self.histories.get(session_id, [])
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

    def clear(self, session_id: str):
        self.histories[session_id] = []

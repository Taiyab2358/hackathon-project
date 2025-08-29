from flask import Flask, request, Response, render_template, session
import subprocess
import uuid
from history import ChatHistory

app = Flask(__name__)
app.secret_key = "super_secret_key_for_sessions"  # change in production

DEFAULT_MODEL = "qwen2.5:1.5b"
chat_history = ChatHistory()


def get_session_id():
    """Assigns a unique session ID to each user if not already set."""
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]


def stream_ollama(prompt: str, model: str = DEFAULT_MODEL):
    try:
        process = subprocess.Popen(
            ["ollama", "run", model, prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        for line in iter(process.stdout.readline, ''):
            if line.strip():
                yield line.strip()
        process.stdout.close()
        process.wait()
    except Exception as e:
        yield f"Error: {str(e)}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    session_id = get_session_id()

    if not user_input:
        return Response("No input provided", status=400)

    chat_history.add_user_message(session_id, user_input)

    def generate():
        full_prompt = chat_history.get_formatted_history(session_id)
        bot_response = ""
        try:
            for chunk in stream_ollama(full_prompt, DEFAULT_MODEL):
                bot_response += chunk + " "
                yield f"data: {chunk}\n\n"
            chat_history.add_bot_message(session_id, bot_response.strip())
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/reset", methods=["POST"])
def reset():
    session_id = get_session_id()
    chat_history.clear(session_id)
    return {"status": f"cleared for session {session_id}"}


if __name__ == "__main__":
    app.run(debug=True, threaded=True)

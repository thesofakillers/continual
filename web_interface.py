from flask import Flask, request, jsonify, send_file
from harness import model_harness

app = Flask(__name__)


@app.route("/")
def index():
    return send_file("chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        messages = data.get("messages", [])

        if not messages:
            return jsonify({"error": "No messages provided"}), 400

        response = model_harness(messages)

        return jsonify({"response": response})

    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9999)


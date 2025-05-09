from flask import Flask, request, jsonify
from tools.search import run_duckduckgo
import requests
import os

app = Flask(__name__)

OLLAMA_URL = os.getenv("OLLAMA_BACKEND", "http://192.168.1.8:11434")

@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json()
    user_prompt = data.get("prompt", "")

    # === AGENT LOGIC: Intercept 'search' intent ===
    if "search" in user_prompt.lower() or "look up" in user_prompt.lower():
        query = user_prompt.lower().replace("search", "").replace("look up", "").strip()
        results = run_duckduckgo(query)
        return jsonify({"response": results})

    # === Otherwise, forward to Ollama ===
    try:
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=data)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/tags", methods=["GET"])
def list_models():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def health():
    return "AI-Agent proxy up!", 200

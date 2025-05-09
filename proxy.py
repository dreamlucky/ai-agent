from flask import Flask, request, jsonify, Response, stream_with_context
from tools.search import run_duckduckgo
import requests
import os
import json

app = Flask(__name__)

OLLAMA_URL = os.getenv("OLLAMA_BACKEND", "http://192.168.1.8:11434")

@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json()
    print("[DEBUG] Direct generate data:", data)

    user_prompt = data.get("prompt", "")
    model = data.get("model", "qwen3:30b-a3b")
    use_stream = data.get("stream", True)

    # === AGENT TOOL LOGIC: Intercept search ===
    if "search" in user_prompt.lower() or "look up" in user_prompt.lower():
        query = user_prompt.lower().replace("search", "").replace("look up", "").strip()
        results = run_duckduckgo(query)

        if use_stream:
            def stream_one_result():
                yield f'data: {json.dumps({"response": results})}\n\n'
                yield 'data: [DONE]\n\n'

            return Response(stream_with_context(stream_one_result()), mimetype='text/event-stream')
        else:
            return jsonify({"response": results})

    # === Streamed response ===
    ollama_payload = {
        "model": model,
        "prompt": user_prompt,
        "stream": use_stream
    }

    try:
        upstream = requests.post(f"{OLLAMA_URL}/api/generate", json=ollama_payload, stream=use_stream)

        if not use_stream:
            return jsonify(upstream.json()), upstream.status_code

        def stream_response():
            for line in upstream.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        yield f'data: {json.dumps(chunk)}\n\n'
                    except Exception as e:
                        print("[STREAM ERROR]", e)

        return Response(stream_with_context(stream_response()), mimetype='text/event-stream')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    print("[DEBUG] OpenAI-style chat data:", data)

    messages = data.get("messages", [])
    model = data.get("model", "qwen3:30b-a3b")
    prompt = "\n".join([msg.get("content", "") for msg in messages])

    ollama_payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }

    try:
        upstream = requests.post(f"{OLLAMA_URL}/api/generate", json=ollama_payload, stream=True)

        def generate_stream():
            # Initial role (required by Open WebUI)
            yield 'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n'

            for line in upstream.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        content = chunk.get("response", "")
                        if content:
                            yield f'data: {json.dumps({"choices": [{"delta": {"content": content}}]})}\n\n'
                    except Exception as e:
                        print("[STREAM ERROR]", e)

            # Finish sequence
            yield 'data: {"choices": [{"delta": {},"finish_reason": "stop"}]}\n\n'

        return Response(stream_with_context(generate_stream()), mimetype='text/event-stream')

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/tags", methods=["GET"])
def list_models():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags")
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/version", methods=["GET"])
def version():
    return jsonify({"version": "0.1.0"}), 200

@app.route("/")
def health():
    return "AI-Agent proxy up!", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

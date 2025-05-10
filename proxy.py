from flask import Flask, request, jsonify, Response, stream_with_context
import requests
import os
import json
import time
from tools.search import run_duckduckgo

app = Flask(__name__)

OLLAMA_URL = os.getenv("OLLAMA_BACKEND", "http://localhost:11434")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    messages = data.get("messages", [])
    model = data.get("model", "qwen3:30b-a3b")
    stream = data.get("stream", False)

    prompt = "\n".join([msg.get("content", "") for msg in messages])

    # DuckDuckGo intercept
    if "search" in prompt.lower() or "look up" in prompt.lower():
        query = prompt.lower().replace("search", "").replace("look up", "").strip()
        results = run_duckduckgo(query)

        if stream:
            def stream_one_result():
                yield f'data: {json.dumps({"response": results})}\n\n'
                yield 'data: [DONE]\n\n'
            return Response(stream_with_context(stream_one_result()), mimetype='text/event-stream')
        else:
            return jsonify({"response": results})

    ollama_payload = {
        "model": model,
        "prompt": prompt,
        "stream": stream
    }

    try:
        upstream = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=ollama_payload,
            stream=stream,
            timeout=300
        )

        if upstream.status_code != 200:
            print("[ERROR] Ollama returned error:", upstream.text)
            return jsonify({"error": f"Ollama error: {upstream.text}"}), upstream.status_code

        if not stream:
            try:
                raw = upstream.json()
                return jsonify({
                    "id": "chatcmpl-ollama",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": raw.get("response", "")
                        },
                        "finish_reason": "stop"
                    }]
                })
            except Exception as e:
                print("[ERROR] Failed to parse non-streaming JSON:", e)
                print("[DEBUG] Upstream raw:", upstream.text)
                return jsonify({"error": "Invalid response from Ollama"}), 500

        def stream_response():
            yield 'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            for line in upstream.iter_lines():
                if not line:
                    continue
                decoded = line.decode("utf-8").strip()
                if decoded == "[DONE]":
                    yield 'data: {"choices":[{"delta":{}}],"finish_reason":"stop"}\n\n'
                    break
                if decoded.startswith("data: "):
                    decoded = decoded[6:]
                try:
                    parsed = json.loads(decoded)
                    content = parsed.get("response", "")
                    if content:
                        yield f'data: {json.dumps({"choices": [{"delta": {"content": content}}]})}\n\n'
                except Exception as e:
                    print("[STREAM PARSE ERROR]", decoded, e)

        return Response(stream_response(), content_type="text/event-stream")

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
        print("[ERROR] /api/tags failed:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/api/version", methods=["GET"])
def version():
    return jsonify({"version": "0.1.0"}), 200


@app.route("/")
def health():
    return "AI-Agent proxy up!", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

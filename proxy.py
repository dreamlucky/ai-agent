from flask import Flask, request, jsonify, Response, stream_with_context
import requests
import os
import json
import time
import traceback # Added for more detailed error logging

# --- Mock/Placeholder for run_duckduckgo ---
# Replace this with your actual tools.search import and implementation
# from tools.search import run_duckduckgo
def run_duckduckgo(query: str) -> str:
    """
    Placeholder for your DuckDuckGo search function.
    It should return a string or a serializable structure.
    """
    print(f"[INFO] Mock DuckDuckGo search for: {query}")
    # Simulate some search results as a string
    return f"Search results for '{query}':\n1. Mock Result Alpha for {query}.\n2. Mock Result Beta for {query}."
# --- End Mock ---

app = Flask(__name__)

OLLAMA_URL = os.getenv("OLLAMA_BACKEND", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen3:30b-a3b") # Made default model configurable

@app.route("/api/chat", methods=["POST"])
def chat_proxy():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        messages = data.get("messages", [])
        model = data.get("model", DEFAULT_MODEL)
        stream = data.get("stream", False)

        if not messages:
            return jsonify({"error": "Missing 'messages' field"}), 400

        # Construct prompt for Ollama's /api/generate
        # This flattens history. For better chat, consider Ollama's /api/chat if available
        # and modify payload accordingly.
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        # Add a final marker for the assistant to start generation, if desired by the model
        prompt = "\n".join(prompt_parts) + "\nAssistant:"


        # --- DuckDuckGo Tool Intercept ---
        # Basic keyword detection in the last user message.
        # For more complex tooling, a dedicated tool dispatcher and parser would be better.
        last_message_content = messages[-1].get("content", "").lower() if messages else ""
        is_search_request = False
        search_query = ""

        if "search for " in last_message_content:
            search_query = last_message_content.split("search for ", 1)[1].strip()
            is_search_request = True
        elif "look up " in last_message_content:
            search_query = last_message_content.split("look up ", 1)[1].strip()
            is_search_request = True
        
        if is_search_request and search_query:
            print(f"[INFO] Intercepted search query: '{search_query}'")
            search_results_text = run_duckduckgo(search_query)
            
            tool_model_name = f"duckduckgo_tool/{model}" # Indicate tool usage with base model

            if stream:
                def stream_search_response():
                    # Initial delta (optional, but can set role)
                    # yield f'data: {json.dumps({"id": f"chatcmpl-ddg-{time.time_ns()}", "object": "chat.completion.chunk", "created": int(time.time()), "model": tool_model_name, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})}\n\n'
                    
                    # Content delta with search results
                    content_payload = {
                        "id": f"chatcmpl-ddg-{time.time_ns()}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": tool_model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": search_results_text}, # Send all results as one content chunk
                            "finish_reason": None 
                        }]
                    }
                    yield f'data: {json.dumps(content_payload)}\n\n'
                    
                    # Final delta indicating the turn is over (e.g., due to tool use)
                    final_payload = {
                        "id": f"chatcmpl-ddg-{time.time_ns()}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": tool_model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {}, # Empty delta
                            "finish_reason": "stop" # Or "tool_calls" if your client handles that
                        }]
                    }
                    yield f'data: {json.dumps(final_payload)}\n\n'
                    yield 'data: [DONE]\n\n'
                return Response(stream_with_context(stream_search_response()), mimetype='text/event-stream')
            else: # Non-streaming search result
                return jsonify({
                    "id": f"chatcmpl-ddg-{time.time_ns()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": tool_model_name,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": search_results_text
                        },
                        "finish_reason": "stop" # Or "tool_calls"
                    }],
                    # "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0} # Placeholder
                })
        # --- End DuckDuckGo Tool Intercept ---


        # --- Ollama Backend Call ---
        ollama_payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            # "options": {"temperature": 0.7} # Example option
            # If using Ollama's /api/chat endpoint, the payload structure would be different:
            # "messages": messages 
        }
        
        ollama_api_url = f"{OLLAMA_URL}/api/generate" # Using /api/generate
        print(f"[INFO] Relaying to Ollama ({ollama_api_url}) with payload: {json.dumps(ollama_payload, indent=2)}")

        upstream_response = requests.post(
            ollama_api_url,
            json=ollama_payload,
            stream=stream,
            timeout=300  # 5 minutes for potentially long generations
        )
        upstream_response.raise_for_status() # Raise HTTPError for 4xx/5xx status codes

        if not stream: # Non-streaming response from Ollama
            ollama_json = upstream_response.json()
            generated_content = ollama_json.get("response", "").strip()
            # Ollama's non-streaming /api/generate also returns model, created_at, done, context, and timing fields
            
            # Format for OpenAI compatibility
            return jsonify({
                "id": f"chatcmpl-ollama-{time.time_ns()}",
                "object": "chat.completion",
                "created": int(time.time()), # Standard Unix timestamp
                "model": ollama_json.get("model", model), # Use model from Ollama if available
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_content
                    },
                    "finish_reason": "stop" # Ollama's /generate implies stop when done
                }],
                # "usage": { # You might parse these from Ollama's detailed response if needed
                #     "prompt_tokens": ollama_json.get("prompt_eval_count", 0),
                #     "completion_tokens": ollama_json.get("eval_count", 0),
                #     "total_tokens": ollama_json.get("prompt_eval_count", 0) + ollama_json.get("eval_count", 0)
                # }
            })
        else: # Streaming response from Ollama
            def generate_openai_formatted_stream():
                # Send initial delta establishing role (OpenAI client convention)
                initial_created_time = int(time.time())
                initial_chunk_id = f"chatcmpl-ollama-stream-{time.time_ns()}"
                yield f'data: {json.dumps({"id": initial_chunk_id, "object": "chat.completion.chunk", "created": initial_created_time, "model": model, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})}\n\n'
                
                # For debugging, to see the full sequence of lines from Ollama if issues arise
                # ollama_raw_stream_lines_for_debug = []

                for line_bytes in upstream_response.iter_lines():
                    if not line_bytes: # Skip empty keep-alive lines
                        continue
                    
                    line_str = line_bytes.decode('utf-8')
                    # ollama_raw_stream_lines_for_debug.append(line_str)

                    try:
                        ollama_chunk = json.loads(line_str)
                        # Expected fields in Ollama /api/generate stream chunks:
                        # "model", "created_at", "response" (the text chunk), "done" (boolean)
                        # Final chunk (done=true) may also have: "context", "total_duration", "load_duration", etc.
                        
                        content_text = ollama_chunk.get("response", "")
                        is_done = ollama_chunk.get("done", False)
                        
                        current_chunk_id = f"chatcmpl-ollama-stream-{time.time_ns()}"
                        current_created_time = int(time.time()) # Can also use ollama_chunk.get("created_at") if parsed
                        
                        delta_payload = {}
                        if content_text: # Only include "content" if there's text
                            delta_payload["content"] = content_text
                        
                        choice_item = {
                            "index": 0,
                            "delta": delta_payload,
                            "finish_reason": None
                        }

                        if is_done:
                            choice_item["finish_reason"] = "stop"
                            # Optionally, include usage from the final ollama_chunk if available and desired
                            # usage_stats_from_ollama_final_chunk = { ... }

                        sse_event_data = {
                            "id": current_chunk_id,
                            "object": "chat.completion.chunk",
                            "created": current_created_time,
                            "model": ollama_chunk.get("model", model), # Use model from chunk
                            "choices": [choice_item]
                            # if is_done and usage_stats_from_ollama_final_chunk:
                            #     sse_event_data["usage"] = usage_stats_from_ollama_final_chunk
                        }
                        yield f'data: {json.dumps(sse_event_data)}\n\n'

                        if is_done:
                            print(f"[INFO] Ollama stream part marked done. Final chunk: {ollama_chunk}")
                            break # Exit the loop as Ollama has finished

                    except json.JSONDecodeError:
                        # This means Ollama sent a line that wasn't valid JSON.
                        print(f"[WARN] Ollama stream: Skipping non-JSON line: '{line_str}'")
                        continue 
                    except Exception as e_inner:
                        print(f"[ERROR] Ollama stream: Error processing line '{line_str}': {e_inner}")
                        # Depending on severity, you might break or continue. Breaking is safer.
                        break
                
                # After the loop (either completed via "done:true" or broke due to error),
                # send the [DONE] marker for OpenAI-compatible clients.
                yield 'data: [DONE]\n\n'
                # print(f"[DEBUG] Full raw Ollama stream received:\n{''.join(ollama_raw_stream_lines_for_debug)}")
                print("[INFO] SSE stream to client finished.")

            return Response(stream_with_context(generate_openai_formatted_stream()), mimetype='text/event-stream')

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else 500
        error_text = e.response.text if e.response is not None else "Unknown Ollama HTTP Error"
        print(f"[ERROR] Ollama HTTP Error ({status_code}): {error_text}")
        try:
            ollama_error_json = e.response.json()
            error_detail = ollama_error_json.get("error", error_text)
        except ValueError: # Not JSON
            error_detail = error_text
        return jsonify({"error": f"Ollama upstream error: {error_detail}"}), status_code
    except requests.exceptions.RequestException as e: # Catches DNS errors, connection refused, timeouts etc.
        print(f"[ERROR] Ollama Request Exception: {e}")
        return jsonify({"error": f"Cannot connect to Ollama: {str(e)}"}), 503 # Service Unavailable
    except Exception as e:
        print(f"[ERROR] Unhandled exception in /api/chat: {type(e).__name__} - {e}")
        traceback.print_exc() # Print full traceback for debugging
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route("/api/tags", methods=["GET"])
def list_ollama_models():
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10) # Added timeout
        response.raise_for_status() # Raise an exception for HTTP error codes
        # Ollama's /api/tags returns: {"models": [{"name": "model:tag", "modified_at": ..., "size": ...}]}
        # This format is often directly usable by clients that support Ollama.
        # If a specific OpenAI-like model list format is needed by the client, transform here.
        return jsonify(response.json()), response.status_code
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else 500
        error_text = e.response.text if e.response is not None else "Unknown Ollama HTTP Error (tags)"
        print(f"[ERROR] Ollama /api/tags HTTP Error ({status_code}): {error_text}")
        return jsonify({"error": f"Ollama upstream error (tags): {error_text}"}), status_code
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Ollama /api/tags Request Exception: {e}")
        return jsonify({"error": f"Cannot connect to Ollama (tags): {str(e)}"}), 503
    except Exception as e:
        print(f"[ERROR] Unhandled exception in /api/tags: {type(e).__name__} - {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal Server Error (tags): {str(e)}"}), 500

@app.route("/api/version", methods=["GET"])
def proxy_version():
    # This is the version of THIS proxy script
    return jsonify({"version": "0.2.0-updated", "ollama_target": OLLAMA_URL}), 200

@app.route("/")
def health_check():
    # A more robust health check could try a quick HEAD request to OLLAMA_URL
    return "AI Agent Proxy (Ollama compatible) is operational.", 200

if __name__ == "__main__":
    print(f"Starting Flask AI Agent Proxy for Ollama at {OLLAMA_URL}")
    print(f"Default model for /api/chat: {DEFAULT_MODEL}")
    # For production, use a proper WSGI server like Gunicorn or Waitress.
    # Example: gunicorn -w 4 -b 0.0.0.0:5000 your_script_name:app
    app.run(host="0.0.0.0", port=5000, debug=True) # debug=True is for development ONLY
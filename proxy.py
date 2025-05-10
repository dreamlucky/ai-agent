# proxy.py
import os
import json
import time
import traceback
from flask import Flask, request, jsonify, Response, stream_with_context

# LangChain imports
from langchain_ollama import OllamaLLM 
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.agents.output_parsers.react_json_single_input import ReActJsonSingleInputOutputParser

# Import your tools from tools.custom_tools
try:
    from tools.custom_tools import agent_tools
except ImportError:
    print("[ERROR] Could not import 'agent_tools' from 'tools.custom_tools'. Make sure the file exists and is correctly structured.")
    agent_tools = [] 


app = Flask(__name__)

# --- Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BACKEND", "http://localhost:11434") # Base URL for Ollama
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen3:30b-a3b") # Default model for the agent - UPDATED
PROXY_VERSION = "0.3.5-langchain-model-update" # Version of this proxy

# --- LangChain Agent Setup ---
# Initialize LLM
llm = OllamaLLM(base_url=OLLAMA_BASE_URL, model=DEFAULT_MODEL) # Use the imported OllamaLLM directly

# Define a ReAct prompt template for the agent.
react_prompt_template_str = """
Answer the following questions as best you can. You have access to the following tools:
{tools}

Use the following format for your thought process and actions:

Question: the input question you must answer
Thought: you should always think about what to do. Break down the problem if necessary.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer, or I have sufficient information to answer.
Final Answer: the final answer to the original input question.

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

agent_prompt = PromptTemplate.from_template(react_prompt_template_str)

# Create the ReAct agent
if agent_tools: 
    agent = create_react_agent(
        llm=llm,
        tools=agent_tools,
        prompt=agent_prompt,
        output_parser=ReActJsonSingleInputOutputParser()
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=agent_tools,
        verbose=True,
        handle_parsing_errors=True, 
        max_iterations=10,
    )
else:
    agent_executor = None 
    print("[WARN] LangChain Agent Executor not initialized because tools could not be loaded.")

# --- Flask Routes ---

@app.route("/api/chat", methods=["POST"])
def chat_proxy_langchain():
    if not agent_executor:
        return jsonify({"error": "LangChain agent is not available due to tool loading issues."}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        messages = data.get("messages", [])
        stream = data.get("stream", False)

        if not messages:
            return jsonify({"error": "Missing 'messages' field"}), 400

        last_user_message = ""
        if messages and isinstance(messages, list) and len(messages) > 0:
            last_user_message = messages[-1].get("content", "")
        
        if not last_user_message:
            return jsonify({"error": "Empty or invalid user message content"}), 400

        print(f"[INFO] LangChain Agent received query: {last_user_message} (Model: {DEFAULT_MODEL})") # Added model to log

        if stream:
            print("[WARN] Streaming with LangChain agent is experimental in this proxy.")
            
            def generate_agent_stream():
                chunk_id_base = f"chatcmpl-lcagent-stream-{int(time.time_ns())}"
                created_time = int(time.time())
                
                yield f'data: {json.dumps({"id": f"{chunk_id_base}-0", "object": "chat.completion.chunk", "created": created_time, "model": DEFAULT_MODEL, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})}\n\n'
                
                full_response_content = ""
                is_final_chunk_sent = False 
                try:
                    for chunk_idx, event_chunk in enumerate(agent_executor.stream({"input": last_user_message})):
                        content_to_stream = None
                        is_current_chunk_final = False

                        if "output" in event_chunk and event_chunk["output"] is not None: 
                            current_output = event_chunk["output"]
                            if full_response_content != current_output:
                                content_to_stream = current_output[len(full_response_content):] if current_output.startswith(full_response_content) else current_output
                                full_response_content = current_output 
                            is_current_chunk_final = True 
                        
                        elif "messages" in event_chunk and event_chunk["messages"]:
                            last_message = event_chunk["messages"][-1]
                            if hasattr(last_message, 'content') and last_message.content:
                                current_chunk_content = last_message.content
                                if current_chunk_content.startswith(full_response_content):
                                    content_to_stream = current_chunk_content[len(full_response_content):]
                                else:
                                    content_to_stream = current_chunk_content 
                                if content_to_stream:
                                   full_response_content += content_to_stream
                        
                        if content_to_stream: 
                            sse_event_data = {
                                "id": f"{chunk_id_base}-{chunk_idx+1}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": DEFAULT_MODEL,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": content_to_stream},
                                    "finish_reason": "stop" if is_current_chunk_final else None
                                }]
                            }
                            yield f'data: {json.dumps(sse_event_data)}\n\n'
                        
                        if is_current_chunk_final:
                            is_final_chunk_sent = True
                            break 
                    
                    if not is_final_chunk_sent and full_response_content:
                         yield f'data: {json.dumps({"id": f"{chunk_id_base}-final", "object": "chat.completion.chunk", "created": int(time.time()), "model": DEFAULT_MODEL, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})}\n\n'
                    elif not full_response_content and not is_final_chunk_sent: 
                         yield f'data: {json.dumps({"id": f"{chunk_id_base}-empty", "object": "chat.completion.chunk", "created": int(time.time()), "model": DEFAULT_MODEL, "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}]})}\n\n'

                except Exception as e_stream:
                    print(f"[ERROR] Error during agent stream: {e_stream}")
                    traceback.print_exc()
                    error_message = f"Error during agent execution: {str(e_stream)}"
                    yield f'data: {json.dumps({"id": f"{chunk_id_base}-error", "object": "chat.completion.chunk", "created": int(time.time()), "model": DEFAULT_MODEL, "choices": [{"index": 0, "delta": {"content": error_message}, "finish_reason": "error"}]})}\n\n'
                
                yield 'data: [DONE]\n\n'
                print("[INFO] SSE stream to client finished (LangChain Agent).")

            return Response(stream_with_context(generate_agent_stream()), mimetype='text/event-stream')
        else: # Non-streaming
            response_payload = agent_executor.invoke({"input": last_user_message})
            agent_final_answer = response_payload.get("output", "Sorry, I could not process your request effectively.")

            return jsonify({
                "id": f"chatcmpl-lcagent-{int(time.time_ns())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": DEFAULT_MODEL,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": agent_final_answer
                    },
                    "finish_reason": "stop" 
                }],
            })

    except Exception as e:
        print(f"[ERROR] Unhandled exception in LangChain chat_proxy: {type(e).__name__} - {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal Server Error in Agent: {str(e)}"}), 500

@app.route("/api/tags", methods=["GET"])
def list_ollama_models():
    try:
        import requests 
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        response.raise_for_status()
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
def proxy_version_route():
    return jsonify({"version": PROXY_VERSION, "ollama_target": OLLAMA_BASE_URL, "default_model": DEFAULT_MODEL}), 200

@app.route("/")
def health_check():
    status = "AI Agent Proxy (LangChain enabled) is operational."
    if not agent_executor:
        status += " WARNING: LangChain agent tools failed to load."
    return status, 200

if __name__ == "__main__":
    print(f"--- Starting Flask AI Agent Proxy (LangChain enabled) ---")
    print(f"Version: {PROXY_VERSION}")
    print(f"Targeting Ollama at: {OLLAMA_BASE_URL}")
    print(f"Default model for agent: {DEFAULT_MODEL}")
    if agent_executor:
        print(f"Agent initialized with tools: {[tool.name for tool in agent_tools]}")
    else:
        print("[CRITICAL] LangChain agent_executor is NOT initialized. Tool-based chat will fail.")
    
    app.run(host="0.0.0.0", port=5000, debug=True)

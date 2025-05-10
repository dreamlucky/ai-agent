# proxy.py
import os
import json
import time
import traceback
from flask import Flask, request, jsonify, Response, stream_with_context
from datetime import datetime, timezone

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
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen3:30b-a3b") # Default model for the agent
PROXY_VERSION = "0.4.0-langchain-stream-refine" # Version of this proxy

# --- LangChain Agent Setup ---
# Initialize LLM
llm = OllamaLLM(base_url=OLLAMA_BASE_URL, model=DEFAULT_MODEL) 

# Define a ReAct prompt template for the agent.
react_prompt_template_str = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

To use a tool, please use the following JSON format exactly:
```json
{{
  "action": "tool name (must be one of [{tool_names}])",
  "action_input": "the input to the tool"
}}
```
After receiving the observation from the tool, continue with your thought process.

If no tool is needed to answer the question, or if you have gathered enough information, provide your answer directly using the following format:
Final Answer: [your final answer here]

IMPORTANT: 
- Only use the JSON format for tool actions.
- For direct answers (when no tool is used or after tool use), ONLY use the "Final Answer:" format.
- Do NOT output any XML-like tags such as <think> or <thought> at any point. Stick strictly to the formats described.

Let's begin!

Question: {input}
Thought: {agent_scratchpad}
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
        max_iterations=10,
        handle_parsing_errors="Check your output and make sure it conforms to the expected JSON format for actions or 'Final Answer: ...' for answers." 
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

        print(f"[INFO] LangChain Agent received query: {last_user_message} (Model: {DEFAULT_MODEL}, Stream: {stream})")

        if stream:
            print("[INFO] Handling stream request for /api/chat in Ollama-compatible format.")
            
            def generate_ollama_compatible_stream():
                accumulated_final_output = "" # To keep track of what part of the final answer has been sent
                is_done_sent = False # Flag to ensure 'done:true' is sent only once at the very end

                try:
                    # Iterate over the LangChain agent's stream
                    for event_chunk in agent_executor.stream({"input": last_user_message}):
                        # The 'output' key in event_chunk usually contains the agent's final answer
                        # or significant intermediate results that are meant to be part of the final answer.
                        if "output" in event_chunk and event_chunk["output"] is not None:
                            full_chunk_output = event_chunk["output"]
                            
                            # Determine the new part of the content to stream
                            # This handles cases where 'output' might be cumulative or repeated
                            new_content_to_stream = ""
                            if full_chunk_output.startswith(accumulated_final_output):
                                new_content_to_stream = full_chunk_output[len(accumulated_final_output):]
                            else:
                                # If it doesn't start with accumulated, it might be a fresh/replaced output
                                new_content_to_stream = full_chunk_output
                                accumulated_final_output = "" # Reset accumulated if output is not a continuation

                            if new_content_to_stream:
                                accumulated_final_output += new_content_to_stream
                                ollama_chunk = {
                                    "model": DEFAULT_MODEL,
                                    "created_at": datetime.now(timezone.utc).isoformat(),
                                    "message": {
                                        "role": "assistant",
                                        "content": new_content_to_stream 
                                    },
                                    "done": False # Not done yet, more might come or a final 'done' chunk
                                }
                                print(f"[DEBUG] Streaming chunk: {json.dumps(ollama_chunk)}")
                                yield f'{json.dumps(ollama_chunk)}\n'
                        
                        # You can add more sophisticated logic here to inspect other parts of event_chunk
                        # if you want to stream intermediate thoughts, but for now, we focus on 'output'.
                        # For example, if 'messages' contains AIMessageChunk, you might stream its content.
                        # However, be careful not to duplicate content already handled by 'output'.

                    # After the loop, send the final 'done: true' message
                    if not is_done_sent:
                        final_done_chunk = {
                            "model": DEFAULT_MODEL,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "message": {
                                "role": "assistant",
                                "content": "" # No new content, just signaling completion
                            },
                            "done": True,
                        }
                        print(f"[DEBUG] Streaming final done chunk: {json.dumps(final_done_chunk)}")
                        yield f'{json.dumps(final_done_chunk)}\n'
                        is_done_sent = True

                except Exception as e_stream:
                    print(f"[ERROR] Error during agent stream generation: {e_stream}")
                    traceback.print_exc()
                    if not is_done_sent: # Ensure done is sent even on error
                        error_response = {
                            "model": DEFAULT_MODEL,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "error": f"Error during agent execution: {str(e_stream)}",
                            "done": True 
                        }
                        yield f'{json.dumps(error_response)}\n'
                
                print("[INFO] Ollama-compatible stream to client finished.")

            return Response(stream_with_context(generate_ollama_compatible_stream()), mimetype='application/x-ndjson')
        
        else: # Non-streaming response
            print("[INFO] Handling non-stream request for /api/chat in Ollama-compatible format.")
            response_payload = agent_executor.invoke({"input": last_user_message})
            agent_final_answer = response_payload.get("output", "Sorry, I could not process your request effectively.")

            ollama_response = {
                "model": DEFAULT_MODEL,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "message": {
                    "role": "assistant",
                    "content": agent_final_answer
                },
                "done": True,
            }
            return jsonify(ollama_response)

    except Exception as e:
        print(f"[ERROR] Unhandled exception in LangChain chat_proxy: {type(e).__name__} - {e}")
        traceback.print_exc()
        return jsonify({
            "error": f"Internal Server Error in Agent: {str(e)}",
            "model": DEFAULT_MODEL, 
            "done": True 
            }), 500

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
    status = "AI Agent Proxy (LangChain enabled, Ollama-style stream) is operational."
    if not agent_executor:
        status += " WARNING: LangChain agent tools failed to load."
    return status, 200

if __name__ == "__main__":
    print(f"--- Starting Flask AI Agent Proxy (LangChain enabled, Ollama-style stream) ---")
    print(f"Version: {PROXY_VERSION}")
    print(f"Targeting Ollama at: {OLLAMA_BASE_URL}")
    print(f"Default model for agent: {DEFAULT_MODEL}")
    if agent_executor:
        print(f"Agent initialized with tools: {[tool.name for tool in agent_tools]}")
    else:
        print("[CRITICAL] LangChain agent_executor is NOT initialized. Tool-based chat will fail.")
    
    app.run(host="0.0.0.0", port=5000, debug=True)

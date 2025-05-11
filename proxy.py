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
# from langchain_core.messages import AIMessage, HumanMessage # Not strictly needed for current history string format

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
PROXY_VERSION = "0.5.1-langchain-enhanced-memory-prompt" # Version of this proxy
MEMORY_WINDOW_SIZE = os.getenv("MEMORY_WINDOW_SIZE", "5") # Number of past interactions (user + AI)

# --- LangChain Agent Setup ---
# Initialize LLM
llm = OllamaLLM(base_url=OLLAMA_BASE_URL, model=DEFAULT_MODEL) 

# Define a ReAct prompt template for the agent, now with more explicit history instructions.
react_prompt_template_str = """
You are a helpful assistant. Your primary goal is to answer the user's NEW INPUT.
Before answering, CAREFULLY REVIEW the PREVIOUS CONVERSATION below to understand the context, especially if the NEW INPUT seems to be a follow-up question.
Use the PREVIOUS CONVERSATION to inform your response to the NEW INPUT.

You have access to the following tools:
{tools}

To use a tool, please use the following JSON format exactly:
```json
{{
  "action": "tool name (must be one of [{tool_names}])",
  "action_input": "the input to the tool"
}}
```
After receiving the observation from the tool, continue with your thought process, remembering the PREVIOUS CONVERSATION and the NEW INPUT.

If no tool is needed to answer the NEW INPUT, or if you have gathered enough information (from tools or PREVIOUS CONVERSATION), provide your answer directly using the following format:
Final Answer: [your final answer here]

IMPORTANT: 
- Only use the JSON format for tool actions.
- For direct answers, ONLY use the "Final Answer:" format.
- Do NOT output any XML-like tags such as <think> or <thought> at any point. Stick strictly to the formats described.

PREVIOUS CONVERSATION:
{chat_history}

NEW INPUT:
Question: {input}

Now, begin your thought process.
Thought: {agent_scratchpad}
"""

agent_prompt = PromptTemplate.from_template(react_prompt_template_str).partial(
    tools="\n".join([f"{tool.name}: {tool.description}" for tool in agent_tools]),
    tool_names=", ".join([tool.name for tool in agent_tools]),
)

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
        handle_parsing_errors="Parsing Error: Check your output and make sure it conforms to the expected JSON format for actions or 'Final Answer: ...' for answers. Do not use XML-like tags."
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

        all_messages_from_request = data.get("messages", [])
        stream = data.get("stream", False) 

        if not all_messages_from_request:
            return jsonify({"error": "Missing 'messages' field"}), 400

        last_user_message_content = all_messages_from_request[-1].get("content", "")
        if not last_user_message_content:
            return jsonify({"error": "Empty user message content"}), 400

        chat_history_for_prompt = []
        history_messages_to_consider = all_messages_from_request[:-1] 
        
        try:
            window_size = int(MEMORY_WINDOW_SIZE)
        except ValueError:
            print(f"[WARN] Invalid MEMORY_WINDOW_SIZE value: '{MEMORY_WINDOW_SIZE}'. Defaulting to 5.")
            window_size = 5
            
        start_index = max(0, len(history_messages_to_consider) - (window_size * 2))
        relevant_history = history_messages_to_consider[start_index:]

        for msg in relevant_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                chat_history_for_prompt.append(f"Human: {content}")
            elif role == "assistant": # Also consider 'ai' or other assistant roles if OpenWebUI uses them
                chat_history_for_prompt.append(f"AI: {content}")
        
        chat_history_str = "\n".join(chat_history_for_prompt)
        if not chat_history_str: # Handle case of no prior history
            chat_history_str = "No previous conversation."


        print(f"[INFO] LangChain Agent received query: '{last_user_message_content}' (Model: {DEFAULT_MODEL}, Stream: {stream})")
        if relevant_history: # Only print if there's actual history being passed
            print(f"[DEBUG] Passing chat history to agent:\n{chat_history_str}")
        else:
            print("[DEBUG] No prior chat history being passed to agent for this turn.")


        agent_input = {
            "input": last_user_message_content,
            "chat_history": chat_history_str 
        }


        if stream:
            print("[INFO] Handling stream request for /api/chat in Ollama-compatible format.")
            
            def generate_ollama_compatible_stream():
                accumulated_final_output = "" 
                is_done_sent = False 
                try:
                    for event_chunk in agent_executor.stream(agent_input): 
                        new_content_to_stream = ""
                        
                        if "output" in event_chunk and event_chunk["output"] is not None:
                            full_chunk_output = event_chunk["output"]
                            if full_chunk_output.startswith(accumulated_final_output):
                                new_content_to_stream = full_chunk_output[len(accumulated_final_output):]
                            else:
                                new_content_to_stream = full_chunk_output
                                accumulated_final_output = "" 
                            
                            if new_content_to_stream: 
                                accumulated_final_output += new_content_to_stream
                            
                        if new_content_to_stream:
                            ollama_chunk = {
                                "model": DEFAULT_MODEL,
                                "created_at": datetime.now(timezone.utc).isoformat(),
                                "message": { "role": "assistant", "content": new_content_to_stream },
                                "done": False 
                            }
                            print(f"[DEBUG] Streaming chunk: {json.dumps(ollama_chunk)}")
                            yield f'{json.dumps(ollama_chunk)}\n'
                        
                    if not is_done_sent:
                        final_done_chunk = {
                            "model": DEFAULT_MODEL,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                            "message": { "role": "assistant", "content": "" }, 
                            "done": True,
                        }
                        print(f"[DEBUG] Streaming final done chunk: {json.dumps(final_done_chunk)}")
                        yield f'{json.dumps(final_done_chunk)}\n'
                        is_done_sent = True

                except Exception as e_stream:
                    print(f"[ERROR] Error during agent stream generation: {e_stream}")
                    traceback.print_exc()
                    if not is_done_sent: 
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
            response_payload = agent_executor.invoke(agent_input) 
            agent_final_answer = response_payload.get("output", "Sorry, I could not process your request effectively.")

            ollama_response = {
                "model": DEFAULT_MODEL,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "message": { "role": "assistant", "content": agent_final_answer },
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
    print(f"Memory window size (interactions): {MEMORY_WINDOW_SIZE}")
    if agent_executor:
        print(f"Agent initialized with tools: {[tool.name for tool in agent_tools]}")
    else:
        print("[CRITICAL] LangChain agent_executor is NOT initialized. Tool-based chat will fail.")
    
    app.run(host="0.0.0.0", port=5000, debug=True)

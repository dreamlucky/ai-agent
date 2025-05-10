# proxy.py
import os
import json
import time
import traceback
from flask import Flask, request, jsonify, Response, stream_with_context

# LangChain imports
from langchain_community.llms import Ollama
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers.react_json_single_input import ReActSingleInputOutputParser


# Import your tools from tools.custom_tools
# Make sure tools/custom_tools.py is in the same directory or accessible in PYTHONPATH
# and contains the 'agent_tools' list and the tool functions.
try:
    from tools.custom_tools import agent_tools
except ImportError:
    print("[ERROR] Could not import 'agent_tools' from 'tools.custom_tools'. Make sure the file exists and is correctly structured.")
    # Fallback to an empty list or a default tool if you want the app to start anyway
    agent_tools = [] 


app = Flask(__name__)

# --- Configuration ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BACKEND", "http://localhost:11434") # Base URL for Ollama
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3:latest") # Default model for the agent
PROXY_VERSION = "0.3.0-langchain" # Version of this proxy

# --- LangChain Agent Setup ---
# Initialize LLM
# The Ollama class in langchain_community.llms expects the base URL.
llm = Ollama(base_url=OLLAMA_BASE_URL, model=DEFAULT_MODEL)

# Define a ReAct prompt template for the agent.
# This template guides the LLM on how to think and use tools.
# The `tools` and `tool_names` variables will be populated by LangChain.
# `input` is the user's query.
# `agent_scratchpad` is where the agent's thoughts and tool usage history are stored.
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
# The agent needs the LLM, the tools, and the prompt.
if agent_tools: # Only create agent if tools were loaded
    agent = create_react_agent(
        llm=llm,
        tools=agent_tools,
        prompt=agent_prompt,
        output_parser=ReActSingleInputOutputParser() # Added for more robust output parsing
    )

    # Create the Agent Executor
    # This runs the agent, executes tools, and manages the interaction loop.
    agent_executor = AgentExecutor(
        agent=agent,
        tools=agent_tools,
        verbose=True,  # Set to True for debugging to see agent's thoughts
        handle_parsing_errors=True, # Useful for debugging LLM output issues
        max_iterations=10, # Prevent overly long loops
        # early_stopping_method="generate" # Stop if LLM generates Final Answer
    )
else:
    agent_executor = None # No agent if tools failed to load
    print("[WARN] LangChain Agent Executor not initialized because tools could not be loaded.")

# --- Flask Routes ---

@app.route("/api/chat", methods=["POST"])
def chat_proxy_langchain():
    """
    Handles chat requests, potentially using the LangChain agent if configured.
    """
    if not agent_executor:
        return jsonify({"error": "LangChain agent is not available due to tool loading issues."}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        messages = data.get("messages", [])
        # model_requested = data.get("model", DEFAULT_MODEL) # Agent uses model set in `llm`
        stream = data.get("stream", False)

        if not messages:
            return jsonify({"error": "Missing 'messages' field"}), 400

        # For LangChain agents, typically pass the latest user query.
        # For chat history, you'd need to integrate memory into the agent_executor.
        last_user_message = ""
        if messages and isinstance(messages, list) and len(messages) > 0:
            last_user_message = messages[-1].get("content", "")
        
        if not last_user_message:
            return jsonify({"error": "Empty or invalid user message content"}), 400

        print(f"[INFO] LangChain Agent received query: {last_user_message}")

        # --- Agent Invocation ---
        # Streaming agent thoughts and actions is complex to map to OpenAI's SSE.
        # This example focuses on non-streaming for the final answer.
        # For streaming, you'd iterate over `agent_executor.stream()` and format chunks.

        if stream:
            # Placeholder for streaming logic.
            # True streaming of agent steps requires careful handling of the `agent_executor.stream()` output.
            # Each event from the stream (tool call, thought, final answer) would need to be formatted.
            print("[WARN] Streaming with LangChain agent is experimental in this proxy.")
            
            def generate_agent_stream():
                # This is a simplified conceptual streaming.
                # Real implementation needs to map agent_executor.stream() events to OpenAI SSE.
                chunk_id_base = f"chatcmpl-lcagent-stream-{int(time.time_ns())}"
                created_time = int(time.time())
                
                # Send an initial delta establishing role (OpenAI client convention)
                yield f'data: {json.dumps({"id": f"{chunk_id_base}-0", "object": "chat.completion.chunk", "created": created_time, "model": DEFAULT_MODEL, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})}\n\n'
                
                full_response_content = ""
                try:
                    # agent_executor.stream returns dictionaries with actions, steps, messages etc.
                    for chunk_idx, event_chunk in enumerate(agent_executor.stream({"input": last_user_message})):
                        # You need to inspect event_chunk to see what it contains (e.g., 'actions', 'steps', 'messages')
                        # and decide what to stream. For this example, we'll just try to get 'output' if it's the final answer.
                        # A more robust solution would parse the 'log' or specific keys from `event_chunk`.
                        
                        # This is a very basic way to get parts of the thought process or final output
                        content_to_stream = None
                        is_final_chunk = False

                        if "output" in event_chunk: # Typically means final answer
                            content_to_stream = event_chunk["output"]
                            if full_response_content: # If we streamed parts before, send only the new part
                                content_to_stream = content_to_stream.replace(full_response_content, "", 1)
                            full_response_content += content_to_stream
                            is_final_chunk = True
                        elif "messages" in event_chunk and event_chunk["messages"]:
                            # messages are usually AIMessage, HumanMessage etc.
                            # Get content from the last AIMessageChunk if available
                            last_message = event_chunk["messages"][-1]
                            if hasattr(last_message, 'content'):
                                content_to_stream = last_message.content
                                if full_response_content:
                                     content_to_stream = content_to_stream.replace(full_response_content, "", 1)
                                full_response_content += content_to_stream
                        # Add more conditions here to stream intermediate thoughts/tool outputs if desired

                        if content_to_stream:
                            sse_event_data = {
                                "id": f"{chunk_id_base}-{chunk_idx+1}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": DEFAULT_MODEL,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": content_to_stream},
                                    "finish_reason": "stop" if is_final_chunk else None
                                }]
                            }
                            yield f'data: {json.dumps(sse_event_data)}\n\n'
                        
                        if is_final_chunk:
                            break # Stop after final output
                    
                    if not full_response_content: # If loop finished with no output (should not happen with ReAct)
                         yield f'data: {json.dumps({"id": f"{chunk_id_base}-final", "object": "chat.completion.chunk", "created": int(time.time()), "model": DEFAULT_MODEL, "choices": [{"index": 0, "delta": {"content": "Agent finished without explicit output."}, "finish_reason": "stop"}]})}\n\n'

                except Exception as e_stream:
                    print(f"[ERROR] Error during agent stream: {e_stream}")
                    traceback.print_exc()
                    error_message = f"Error during agent execution: {str(e_stream)}"
                    yield f'data: {json.dumps({"id": f"{chunk_id_base}-error", "object": "chat.completion.chunk", "created": int(time.time()), "model": DEFAULT_MODEL, "choices": [{"index": 0, "delta": {"content": error_message}, "finish_reason": "error"}]})}\n\n'
                
                yield 'data: [DONE]\n\n'
                print("[INFO] SSE stream to client finished (LangChain Agent).")

            return Response(stream_with_context(generate_agent_stream()), mimetype='text/event-stream')
        else:
            # Non-streaming invocation
            # The `agent_executor.invoke` method takes the input and returns the final result.
            # The input to the ReAct agent is typically a dictionary with the key "input".
            # The `tool_names` and `tools` are usually handled internally by the agent setup.
            response_payload = agent_executor.invoke({"input": last_user_message})
            agent_final_answer = response_payload.get("output", "Sorry, I could not process your request effectively.")

            # Format for OpenAI compatibility (non-streaming)
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
                # "usage": { ... } # Token usage is harder to get accurately from ReAct agents
            })

    except Exception as e:
        print(f"[ERROR] Unhandled exception in LangChain chat_proxy: {type(e).__name__} - {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal Server Error in Agent: {str(e)}"}), 500

@app.route("/api/tags", methods=["GET"])
def list_ollama_models():
    """
    Proxies requests to Ollama's /api/tags to list available models.
    """
    try:
        # Using OLLAMA_BASE_URL which should be like http://ollama_host:port
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        response.raise_for_status()
        # This format is often directly usable by clients like Open WebUI.
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
    """
    Returns the version of this proxy script and the targeted Ollama backend.
    """
    return jsonify({"version": PROXY_VERSION, "ollama_target": OLLAMA_BASE_URL, "default_model": DEFAULT_MODEL}), 200

@app.route("/")
def health_check():
    """
    Basic health check for the proxy.
    """
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
    
    # For production, use a proper WSGI server like Gunicorn or Waitress.
    # Example: gunicorn -w 4 -b 0.0.0.0:5000 proxy:app
    app.run(host="0.0.0.0", port=5000, debug=True) # debug=True is for development ONLY
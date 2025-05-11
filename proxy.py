# proxy.py
import os
import json
import time
import traceback
import sqlite3 # For raw chat history
import chromadb # For vector store
from chromadb.utils import embedding_functions # For Chroma's default or your own
from langchain_community.embeddings import SentenceTransformerEmbeddings # For embedding text

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
OLLAMA_BASE_URL = os.getenv("OLLAMA_BACKEND", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen3:30b-a3b")
PROXY_VERSION = "0.6.0-rag-phase1-init"
MEMORY_WINDOW_SIZE = os.getenv("MEMORY_WINDOW_SIZE", "5")

# --- RAG Configuration ---
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "/app/persistent_data/chat_history.db")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "/app/persistent_data/vector_store/chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
RAG_TOP_K_RESULTS = int(os.getenv("RAG_TOP_K_RESULTS", "3")) # How many relevant docs to retrieve

# --- Initialize RAG Components ---
# 1. SQLite Database for Raw Chat History
def init_sqlite_db():
    """Initializes the SQLite database and creates the chat_history table if it doesn't exist."""
    # Ensure the directory for the SQLite DB exists
    db_dir = os.path.dirname(SQLITE_DB_PATH)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        print(f"[INFO] Created directory for SQLite DB: {db_dir}")

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            turn_id INTEGER NOT NULL,
            role TEXT NOT NULL, CHECK(role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT 
        )
    """)
    conn.commit()
    conn.close()
    print(f"[INFO] SQLite DB for chat history initialized at {SQLITE_DB_PATH}")

# 2. Embedding Model
try:
    print(f"[INFO] Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("[INFO] Embedding model initialized successfully.")
except Exception as e:
    print(f"[ERROR] Failed to initialize embedding model '{EMBEDDING_MODEL_NAME}': {e}")
    embedding_model = None

# 3. ChromaDB Vector Store
try:
    # Ensure the directory for the ChromaDB vector store exists
    if not os.path.exists(VECTOR_STORE_PATH):
        os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        print(f"[INFO] Created directory for Chroma vector store: {VECTOR_STORE_PATH}")

    print(f"[INFO] Initializing ChromaDB persistent client at: {VECTOR_STORE_PATH}")
    # Use a persistent client for ChromaDB
    chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    
    # Use the initialized SentenceTransformerEmbeddings with Chroma
    # If using Chroma's default, it would download its own.
    # For consistency and control, we use our explicitly loaded one.
    # chroma_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    # However, LangChain's Chroma wrapper expects a LangChain Embedding object.
    # We will pass our `embedding_model` when creating LangChain's Chroma wrapper later if needed,
    # or use Chroma client directly with our embeddings.

    COLLECTION_NAME = "chat_interactions"
    print(f"[INFO] Getting or creating Chroma collection: {COLLECTION_NAME}")
    # If using SentenceTransformer directly with Chroma client:
    # collection = chroma_client.get_or_create_collection(
    # name=COLLECTION_NAME, embedding_function=chroma_embedding_function
    # )
    # If using LangChain's Chroma wrapper, collection creation is often handled by the wrapper.
    # For now, we just ensure the client is up. The collection will be used in add/retrieve functions.
    # Let's get or create the collection here to ensure it exists.
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME) # Uses default ef if none specified, or we manage embeddings manually
    print(f"[INFO] ChromaDB client and collection '{COLLECTION_NAME}' initialized.")
except Exception as e:
    print(f"[ERROR] Failed to initialize ChromaDB: {e}")
    chroma_client = None
    collection = None

# Call initialization functions at startup
init_sqlite_db()

# --- LangChain Agent Setup ---
llm = OllamaLLM(base_url=OLLAMA_BASE_URL, model=DEFAULT_MODEL) 

react_prompt_template_str = """
You are a helpful assistant. Your primary goal is to answer the user's NEW INPUT.
Before answering, CAREFULLY REVIEW the PREVIOUS CONVERSATION (short-term window) and any RELEVANT PAST INTERACTIONS (retrieved from long-term memory) to understand the context.
Use all available context to inform your response to the NEW INPUT.

You have access to the following tools:
{tools}

To use a tool, please use the following JSON format exactly:
```json
{{
  "action": "tool name (must be one of [{tool_names}])",
  "action_input": "the input to the tool"
}}
```
After receiving the observation from the tool, continue with your thought process.

If no tool is needed, or if you have gathered enough information, provide your answer directly using the format:
Final Answer: [your final answer here]

IMPORTANT: 
- Only use the JSON format for tool actions.
- For direct answers, ONLY use the "Final Answer:" format.
- Do NOT output any XML-like tags such as <think> or <thought>. Stick strictly to the formats described.

PREVIOUS CONVERSATION (Short-term window):
{chat_history}

RELEVANT PAST INTERACTIONS (Retrieved from Long-Term Memory - if any):
{retrieved_long_term_interactions}

NEW INPUT:
Question: {input}

Now, begin your thought process.
Thought: {agent_scratchpad}
"""

agent_prompt = PromptTemplate.from_template(react_prompt_template_str).partial(
    tools="\n".join([f"{tool.name}: {tool.description}" for tool in agent_tools]),
    tool_names=", ".join([tool.name for tool in agent_tools]),
)

if agent_tools: 
    agent = create_react_agent(llm=llm, tools=agent_tools, prompt=agent_prompt, output_parser=ReActJsonSingleInputOutputParser())
    agent_executor = AgentExecutor(
        agent=agent, tools=agent_tools, verbose=True, max_iterations=10,
        handle_parsing_errors="Parsing Error: Check your output and make sure it conforms to the expected JSON format for actions or 'Final Answer: ...' for answers. Do not use XML-like tags."
    )
else:
    agent_executor = None 
    print("[WARN] LangChain Agent Executor not initialized because tools could not be loaded.")

# --- RAG Helper Functions (Placeholders - to be implemented) ---
def add_interaction_to_db(conversation_id: str, turn_id: int, role: str, content: str, metadata: dict = None):
    """Adds a chat interaction to the SQLite database."""
    if not SQLITE_DB_PATH:
        print("[WARN] SQLITE_DB_PATH not set. Skipping DB log.")
        return
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_history (conversation_id, turn_id, role, content, metadata) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, turn_id, role, content, json.dumps(metadata) if metadata else None)
        )
        conn.commit()
    except sqlite3.Error as e:
        print(f"[ERROR] Failed to write to SQLite DB: {e}")
    finally:
        if conn:
            conn.close()

def add_interaction_to_vector_store(doc_id: str, text_content: str, metadata: dict = None):
    """Embeds and adds a text document to the ChromaDB vector store."""
    if not collection or not embedding_model:
        print("[WARN] Vector store or embedding model not initialized. Skipping vector store add.")
        return
    try:
        print(f"[DEBUG] Embedding document ID: {doc_id} for vector store.")
        embeddings = embedding_model.embed_documents([text_content]) # embed_documents expects a list
        if not embeddings:
            print(f"[WARN] Failed to generate embeddings for doc ID: {doc_id}")
            return

        # Chroma typically expects a list of embeddings, a list of documents (optional if IDs are primary),
        # a list of metadatas, and a list of IDs.
        collection.add(
            ids=[doc_id],
            embeddings=embeddings, # List of embeddings
            metadatas=[metadata if metadata else {}], # List of metadatas
            documents=[text_content] # Optional: store the text directly in Chroma
        )
        print(f"[INFO] Added document ID: {doc_id} to vector store.")
    except Exception as e:
        print(f"[ERROR] Failed to add to vector store for doc ID {doc_id}: {e}")
        traceback.print_exc()


def retrieve_relevant_history(query_text: str, n_results: int = RAG_TOP_K_RESULTS, conv_id: str = None) -> str:
    """Retrieves relevant documents from ChromaDB based on the query text."""
    if not collection or not embedding_model:
        print("[WARN] Vector store or embedding model not initialized. Skipping retrieval.")
        return "No long-term memory available (system not initialized)."
    if not query_text:
        return "No query text provided for retrieval."
        
    try:
        print(f"[DEBUG] Generating query embedding for: '{query_text[:50]}...'")
        query_embedding = embedding_model.embed_query(query_text) # embed_query for single text

        # Optional: Filter by conversation_id if provided and stored in metadata
        where_filter = None
        if conv_id and 'conversation_id' in (collection.peek(limit=1).get('metadatas', [{}])[0] or {}): # Check if metadata has conv_id
             print(f"[DEBUG] Applying filter for conversation_id: {conv_id}")
             where_filter = {"conversation_id": conv_id} # Example filter

        print(f"[DEBUG] Querying vector store. N_results={n_results}, Filter: {where_filter}")
        results = collection.query(
            query_embeddings=[query_embedding], # List of query embeddings
            n_results=n_results,
            # where=where_filter, # Add filter if applicable and metadata is structured for it
            include=['documents', 'metadatas', 'distances'] # Request documents, metadatas, and distances
        )
        
        retrieved_docs_texts = []
        if results and results.get('documents') and results.get('documents')[0]:
            for i, doc_text in enumerate(results['documents'][0]):
                # distance = results['distances'][0][i] if results.get('distances') and results['distances'][0] else 'N/A'
                # metadata = results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] else {}
                # For now, just use the document text.
                # You might format this better, e.g., "Past interaction (similarity X): [text]"
                retrieved_docs_texts.append(f"Retrieved context: {doc_text}")
            
            if retrieved_docs_texts:
                print(f"[INFO] Retrieved {len(retrieved_docs_texts)} relevant documents from vector store.")
                return "\n".join(retrieved_docs_texts)
            else:
                print("[INFO] No relevant documents found in vector store for the query.")
                return "No specific relevant past interactions found in long-term memory."
        else:
            print("[INFO] No results or documents in results from vector store query.")
            return "No specific relevant past interactions found in long-term memory."
            
    except Exception as e:
        print(f"[ERROR] Failed to retrieve from vector store: {e}")
        traceback.print_exc()
        return "Error retrieving from long-term memory."

# --- Flask Routes ---

@app.route("/api/chat", methods=["POST"])
def chat_proxy_langchain():
    if not agent_executor:
        return jsonify({"error": "LangChain agent is not available due to tool loading issues."}), 503

    try:
        data = request.get_json()
        if not data: return jsonify({"error": "Invalid JSON payload"}), 400

        all_messages_from_request = data.get("messages", [])
        stream = data.get("stream", False) 
        # Attempt to get a conversation ID from the request, if OpenWebUI sends one
        # This is hypothetical; OpenWebUI might not send a stable one per chat.
        # If not, we might need to generate/manage one.
        conversation_id = data.get("conversation_id", f"conv_{int(time.time())}") # Fallback to a new ID

        if not all_messages_from_request: return jsonify({"error": "Missing 'messages' field"}), 400

        last_user_message_obj = all_messages_from_request[-1]
        last_user_message_content = last_user_message_obj.get("content", "")
        if not last_user_message_content: return jsonify({"error": "Empty user message content"}), 400

        # --- RAG: Add user message to DB ---
        # Assuming turn_id can be derived from message list length or a dedicated counter
        current_turn_id = len(all_messages_from_request) 
        add_interaction_to_db(conversation_id, current_turn_id, "user", last_user_message_content)
        # For vector store, we might embed user message + some prior context, or just the user message
        # Unique ID for this document in vector store
        user_doc_id = f"{conversation_id}_turn_{current_turn_id}_user" 
        add_interaction_to_vector_store(user_doc_id, f"User asks: {last_user_message_content}", {"conversation_id": conversation_id, "role": "user"})


        # --- Prepare Short-Term History for Prompt ---
        chat_history_for_prompt_str = []
        history_messages_to_consider = all_messages_from_request[:-1] 
        try: window_size = int(MEMORY_WINDOW_SIZE)
        except ValueError: window_size = 5
        start_index = max(0, len(history_messages_to_consider) - (window_size * 2))
        relevant_short_term_history = history_messages_to_consider[start_index:]
        for msg in relevant_short_term_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            chat_history_for_prompt_str.append(f"{role.capitalize()}: {content}")
        chat_history_str = "\n".join(chat_history_for_prompt_str) if chat_history_for_prompt_str else "No recent conversation history."

        # --- RAG: Retrieve Relevant Long-Term History ---
        # Use current user message and maybe some short-term history as query for RAG
        query_for_rag = last_user_message_content 
        # If short-term history exists, prepend it to make the RAG query more contextual
        if chat_history_str != "No recent conversation history.":
             query_for_rag = f"Recent context:\n{chat_history_str}\n\nUser's current question: {last_user_message_content}"
        
        retrieved_long_term_interactions = retrieve_relevant_history(query_for_rag, conv_id=conversation_id)


        print(f"[INFO] LangChain Agent received query: '{last_user_message_content}' (Model: {DEFAULT_MODEL}, Stream: {stream})")
        if relevant_short_term_history: print(f"[DEBUG] Passing short-term history to agent:\n{chat_history_str}")
        if retrieved_long_term_interactions not in ["No long-term memory available (system not initialized).", "No specific relevant past interactions found in long-term memory.", "Error retrieving from long-term memory."]:
            print(f"[DEBUG] Passing retrieved long-term interactions to agent:\n{retrieved_long_term_interactions}")


        agent_input = {
            "input": last_user_message_content,
            "chat_history": chat_history_str,
            "retrieved_long_term_interactions": retrieved_long_term_interactions
        }

        if stream:
            # (Streaming logic remains largely the same as version 0.5.1, just ensure agent_input is used)
            print("[INFO] Handling stream request for /api/chat in Ollama-compatible format.")
            def generate_ollama_compatible_stream():
                # ... (Keep the streaming logic from 0.5.1, ensuring it uses `agent_input`)
                # For brevity, not repeating the full stream logic here.
                # Key is that `agent_executor.stream(agent_input)` is called.
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
                            if new_content_to_stream: accumulated_final_output += new_content_to_stream
                        if new_content_to_stream:
                            ollama_chunk = {
                                "model": DEFAULT_MODEL, "created_at": datetime.now(timezone.utc).isoformat(),
                                "message": { "role": "assistant", "content": new_content_to_stream }, "done": False 
                            }
                            yield f'{json.dumps(ollama_chunk)}\n'
                    if not is_done_sent:
                        final_done_chunk = {
                            "model": DEFAULT_MODEL, "created_at": datetime.now(timezone.utc).isoformat(),
                            "message": { "role": "assistant", "content": "" }, "done": True,
                        }
                        yield f'{json.dumps(final_done_chunk)}\n'
                        is_done_sent = True
                except Exception as e_stream:
                    print(f"[ERROR] Error during agent stream generation: {e_stream}")
                    traceback.print_exc()
                    if not is_done_sent: 
                        error_response = {
                            "model": DEFAULT_MODEL, "created_at": datetime.now(timezone.utc).isoformat(),
                            "error": f"Error during agent execution: {str(e_stream)}", "done": True 
                        }
                        yield f'{json.dumps(error_response)}\n'
                print("[INFO] Ollama-compatible stream to client finished.")
            return Response(stream_with_context(generate_ollama_compatible_stream()), mimetype='application/x-ndjson')
        
        else: # Non-streaming response
            print("[INFO] Handling non-stream request for /api/chat in Ollama-compatible format.")
            response_payload = agent_executor.invoke(agent_input) 
            agent_final_answer = response_payload.get("output", "Sorry, I could not process your request effectively.")

            # --- RAG: Add assistant response to DB & Vector Store ---
            add_interaction_to_db(conversation_id, current_turn_id + 1, "assistant", agent_final_answer) # turn_id +1 for assistant
            assistant_doc_id = f"{conversation_id}_turn_{current_turn_id + 1}_assistant"
            add_interaction_to_vector_store(assistant_doc_id, f"Assistant responds: {agent_final_answer}", {"conversation_id": conversation_id, "role": "assistant"})

            ollama_response = {
                "model": DEFAULT_MODEL, "created_at": datetime.now(timezone.utc).isoformat(),
                "message": { "role": "assistant", "content": agent_final_answer }, "done": True,
            }
            return jsonify(ollama_response)

    except Exception as e:
        print(f"[ERROR] Unhandled exception in LangChain chat_proxy: {type(e).__name__} - {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal Server Error in Agent: {str(e)}", "model": DEFAULT_MODEL, "done": True }), 500

# ... (rest of the Flask routes: /api/tags, /api/version, / remain the same) ...
@app.route("/api/tags", methods=["GET"])
def list_ollama_models():
    try:
        import requests 
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        response.raise_for_status()
        return jsonify(response.json()), response.status_code
    except Exception as e: # Simplified error handling for brevity
        print(f"[ERROR] /api/tags error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/version", methods=["GET"])
def proxy_version_route():
    return jsonify({"version": PROXY_VERSION, "ollama_target": OLLAMA_BASE_URL, "default_model": DEFAULT_MODEL}), 200

@app.route("/")
def health_check():
    status = "AI Agent Proxy (RAG Phase 1 Initialized) is operational."
    if not agent_executor: status += " WARNING: LangChain agent tools failed to load."
    if not embedding_model: status += " WARNING: Embedding model failed to load."
    if not chroma_client or not collection: status += " WARNING: ChromaDB vector store failed to load."
    return status, 200


if __name__ == "__main__":
    print(f"--- Starting Flask AI Agent Proxy (RAG Phase 1 Initialized) ---")
    print(f"Version: {PROXY_VERSION}")
    print(f"Targeting Ollama at: {OLLAMA_BASE_URL}")
    print(f"Default model for agent: {DEFAULT_MODEL}")
    print(f"Memory window size (interactions): {MEMORY_WINDOW_SIZE}")
    print(f"SQLite DB Path: {SQLITE_DB_PATH}")
    print(f"Vector Store Path: {VECTOR_STORE_PATH}")
    print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
    if agent_executor: print(f"Agent initialized with tools: {[tool.name for tool in agent_tools]}")
    else: print("[CRITICAL] LangChain agent_executor is NOT initialized.")
    
    app.run(host="0.0.0.0", port=5000, debug=True)

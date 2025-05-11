# proxy.py
import os
import json
import time
import traceback
import sqlite3 # For raw chat history
import chromadb # For vector store

# Corrected import for embeddings to address deprecation
from langchain_huggingface import HuggingFaceEmbeddings 

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
PROXY_VERSION = "0.6.6-rag-db-pre-check" # Version of this proxy
MEMORY_WINDOW_SIZE = os.getenv("MEMORY_WINDOW_SIZE", "5")

# --- RAG Configuration ---
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "/app/persistent_data/chat_history.db")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "/app/persistent_data/vector_store/chroma_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
RAG_TOP_K_RESULTS = int(os.getenv("RAG_TOP_K_RESULTS", "3")) 

# --- Initialize RAG Components ---
# 1. SQLite Database for Raw Chat History
def init_sqlite_db():
    """Initializes the SQLite database and creates the chat_history table if it doesn't exist."""
    db_dir = os.path.dirname(SQLITE_DB_PATH)
    if not os.path.exists(db_dir):
        try:
            os.makedirs(db_dir, exist_ok=True)
            print(f"[INFO] Created directory for SQLite DB: {db_dir}")
        except OSError as e:
            print(f"[ERROR] Failed to create directory {db_dir}: {e}")
            traceback.print_exc()
            return # Stop if directory creation fails

    # Check if DB file exists *before* trying to connect or create
    if os.path.exists(SQLITE_DB_PATH):
        print(f"[DEBUG] Pre-check: SQLite DB file '{SQLITE_DB_PATH}' ALREADY EXISTS before connect attempt.")
    else:
        print(f"[DEBUG] Pre-check: SQLite DB file '{SQLITE_DB_PATH}' DOES NOT EXIST before connect attempt. Will be created by connect().")

    conn = None 
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH) # This will create the file if it doesn't exist
        cursor = conn.cursor()
        
        # Check if the table already exists and what its schema is (for debugging)
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='chat_history';")
        existing_table_schema = cursor.fetchone()
        if existing_table_schema:
            print(f"[DEBUG] Existing 'chat_history' table schema: {existing_table_schema[0]}")
        else:
            print("[DEBUG] 'chat_history' table does not exist yet. Will be created.")

        # Create table if it doesn't exist with the correct schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL, 
                conversation_id TEXT NOT NULL,
                turn_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT 
            )
        """)
        conn.commit()
        print(f"[INFO] SQLite DB for chat history initialized/verified at {SQLITE_DB_PATH}")
    except sqlite3.Error as e:
        print(f"[ERROR] SQLite initialization error: {e}")
        traceback.print_exc()
    finally:
        if conn:
            conn.close()

# 2. Embedding Model
embedding_model = None
try:
    print(f"[INFO] Initializing embedding model: {EMBEDDING_MODEL_NAME} using HuggingFaceEmbeddings")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("[INFO] Embedding model initialized successfully.")
except Exception as e:
    print(f"[ERROR] Failed to initialize embedding model '{EMBEDDING_MODEL_NAME}': {e}")
    traceback.print_exc()


# 3. ChromaDB Vector Store
chroma_client = None
collection = None
try:
    vector_store_dir = os.path.dirname(VECTOR_STORE_PATH) # Get directory for Chroma path
    if not os.path.exists(vector_store_dir): # Ensure parent directory exists
        os.makedirs(vector_store_dir, exist_ok=True)
        print(f"[INFO] Created directory for Chroma vector store parent: {vector_store_dir}")

    print(f"[INFO] Initializing ChromaDB persistent client at: {VECTOR_STORE_PATH}")
    chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    
    COLLECTION_NAME = "user_chat_interactions" 
    print(f"[INFO] Getting or creating Chroma collection: {COLLECTION_NAME}")
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME) 
    print(f"[INFO] ChromaDB client and collection '{COLLECTION_NAME}' initialized.")
except Exception as e:
    print(f"[ERROR] Failed to initialize ChromaDB: {e}")
    traceback.print_exc()

# Call initialization functions at startup
init_sqlite_db()

# --- Add this check right after calling init_sqlite_db() ---
if os.path.exists(SQLITE_DB_PATH):
    print(f"[DEBUG] Post-init check: File '{SQLITE_DB_PATH}' EXISTS (from Python's perspective).")
    try:
        conn_check = sqlite3.connect(SQLITE_DB_PATH)
        cursor_check = conn_check.cursor()
        cursor_check.execute("PRAGMA table_info(chat_history);") # Get column info
        columns = cursor_check.fetchall()
        if columns:
            column_names = [col[1] for col in columns]
            print(f"[DEBUG] Post-init check: 'chat_history' table columns: {column_names}")
            if "user_id" in column_names:
                print(f"[DEBUG] Post-init check: 'user_id' column EXISTS in 'chat_history' table.")
            else:
                print(f"[WARN] Post-init check: 'user_id' column DOES NOT EXIST in 'chat_history' table.")
        else:
            print(f"[WARN] Post-init check: 'chat_history' table IS EMPTY or DOES NOT EXIST in '{SQLITE_DB_PATH}'.")
        conn_check.close()
    except sqlite3.Error as e_check:
        print(f"[WARN] Post-init check: Error accessing '{SQLITE_DB_PATH}': {e_check}")
else:
    print(f"[WARN] Post-init check: File '{SQLITE_DB_PATH}' DOES NOT EXIST (from Python's perspective).")
# --- End of new check ---


# --- LangChain Agent Setup ---
# (The rest of the file remains the same as version 0.6.4)
llm = OllamaLLM(base_url=OLLAMA_BASE_URL, model=DEFAULT_MODEL) 

react_prompt_template_str = """
You are a helpful assistant. Your primary goal is to answer the user's NEW INPUT.
Before answering, CAREFULLY REVIEW the PREVIOUS CONVERSATION (short-term window for this user) and any RELEVANT PAST INTERACTIONS (retrieved from this user's long-term memory) to understand the context.
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

PREVIOUS CONVERSATION (Short-term window for this user):
{chat_history}

RELEVANT PAST INTERACTIONS (Retrieved from this user's Long-Term Memory - if any):
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

# --- RAG Helper Functions ---
def add_interaction_to_db(user_id: str, conversation_id: str, turn_id: int, role: str, content: str, metadata: dict = None):
    """Adds a chat interaction to the SQLite database, now including user_id."""
    if not SQLITE_DB_PATH: print("[WARN] SQLITE_DB_PATH not set. Skipping DB log."); return
    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO chat_history (user_id, conversation_id, turn_id, role, content, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, conversation_id, turn_id, role, content, json.dumps(metadata) if metadata else None)
            )
            conn.commit()
    except sqlite3.Error as e: print(f"[ERROR] Failed to write to SQLite DB for user {user_id}, conv {conversation_id}: {e}")

def add_interaction_to_vector_store(doc_id: str, text_content: str, user_id: str, metadata: dict = None):
    """Embeds and adds a text document to the ChromaDB vector store, ensuring user_id is in metadata."""
    if not collection or not embedding_model:
        print("[WARN] Vector store or embedding model not initialized. Skipping vector store add."); return
    
    if metadata is None: metadata = {}
    metadata["user_id"] = user_id 

    try:
        print(f"[DEBUG] Embedding document ID: {doc_id} for vector store. Metadata: {metadata}")
        embeddings = embedding_model.embed_documents([text_content]) 
        if not embeddings: print(f"[WARN] Failed to generate embeddings for doc ID: {doc_id}"); return
        
        collection.add(ids=[doc_id], embeddings=embeddings, metadatas=[metadata], documents=[text_content])
        print(f"[INFO] Added document ID: {doc_id} to vector store for user {user_id}.")
    except Exception as e:
        print(f"[ERROR] Failed to add to vector store for doc ID {doc_id}, user {user_id}: {e}")
        traceback.print_exc()

def retrieve_relevant_history(query_text: str, user_id: str, n_results: int = RAG_TOP_K_RESULTS) -> str:
    """Retrieves relevant documents from ChromaDB for a specific user."""
    if not collection or not embedding_model:
        print("[WARN] Vector store or embedding model not initialized. Skipping retrieval.")
        return "No long-term memory available (system not initialized)."
    if not query_text: return "No query text provided for retrieval."
    if not user_id: print("[WARN] No user_id provided for retrieval. Cannot scope to user."); return "User ID not provided for memory retrieval."
        
    try:
        print(f"[DEBUG] Generating query embedding for user '{user_id}', query: '{query_text[:50]}...'")
        query_embedding = embedding_model.embed_query(query_text) 
        
        where_filter = {"user_id": user_id} 
        
        print(f"[DEBUG] Querying vector store for user '{user_id}'. N_results={n_results}, Filter: {where_filter}")
        results = collection.query(
            query_embeddings=[query_embedding], 
            n_results=n_results, 
            where=where_filter, 
            include=['documents', 'metadatas'] 
        )
        
        retrieved_docs_texts = []
        if results and results.get('documents') and results.get('documents')[0]:
            for i, doc_text in enumerate(results['documents'][0]):
                doc_metadata = results['metadatas'][0][i] if results.get('metadatas') and results['metadatas'][0] and i < len(results['metadatas'][0]) else {}
                original_conv_id = doc_metadata.get('conversation_id', 'unknown_conv')
                retrieved_docs_texts.append(f"Retrieved context (from your past conv: {original_conv_id}): {doc_text}")
            if retrieved_docs_texts:
                print(f"[INFO] Retrieved {len(retrieved_docs_texts)} relevant documents from vector store for user {user_id}.")
                return "\n".join(retrieved_docs_texts)
        
        print(f"[INFO] No relevant documents found in vector store for user {user_id}.")
        return "No specific relevant past interactions found in your long-term memory."
            
    except Exception as e:
        print(f"[ERROR] Failed to retrieve from vector store for user {user_id}: {e}")
        traceback.print_exc()
        return "Error retrieving from your long-term memory."

# --- Flask Routes ---
@app.route("/api/chat", methods=["POST"])
def chat_proxy_langchain():
    if not agent_executor: return jsonify({"error": "LangChain agent is not available."}), 503
    try:
        data = request.get_json()
        if not data: return jsonify({"error": "Invalid JSON payload"}), 400
        
        all_messages_from_request = data.get("messages", [])
        stream = data.get("stream", False) 
        
        user_id = request.headers.get("X-User-ID", "default_user") 
        print(f"[INFO] Using user_id: {user_id}")

        conversation_id = data.get("conversation_id")
        if not conversation_id and "options" in data and isinstance(data["options"], dict):
            conversation_id = data["options"].get("conversation_id") 
        if not conversation_id and "chat_id" in data: 
            conversation_id = data["chat_id"]
        if not conversation_id:
            conversation_id = f"session_{user_id}_{int(time.time())}_{os.urandom(4).hex()}" 
            print(f"[WARN] No conversation_id from client; generated temporary: {conversation_id}")
        else:
            print(f"[INFO] Using conversation_id from client: {conversation_id}")

        if not all_messages_from_request: return jsonify({"error": "Missing 'messages' field"}), 400
        last_user_message_obj = all_messages_from_request[-1]
        last_user_message_content = last_user_message_obj.get("content", "")
        if not last_user_message_content: return jsonify({"error": "Empty user message content"}), 400

        current_turn_id = len(all_messages_from_request) 
        interaction_metadata = {"user_id": user_id, "conversation_id": conversation_id}
        add_interaction_to_db(user_id, conversation_id, current_turn_id, "user", last_user_message_content, interaction_metadata)
        user_doc_id = f"{user_id}_{conversation_id}_turn_{current_turn_id}_user" 
        add_interaction_to_vector_store(user_doc_id, f"User asks: {last_user_message_content}", user_id, interaction_metadata)

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
        chat_history_str = "\n".join(chat_history_for_prompt_str) if chat_history_for_prompt_str else "No recent conversation history for this user in this session."
        
        query_for_rag = last_user_message_content 
        if chat_history_str != "No recent conversation history for this user in this session.":
             query_for_rag = f"Recent context for this user:\n{chat_history_str}\n\nUser's current question: {last_user_message_content}"
        
        retrieved_long_term_interactions = retrieve_relevant_history(query_for_rag, user_id=user_id) 

        print(f"[INFO] LangChain Agent received query: '{last_user_message_content}' (User: {user_id}, ConvID: {conversation_id}, Model: {DEFAULT_MODEL}, Stream: {stream})")
        if relevant_short_term_history: print(f"[DEBUG] Passing short-term history to agent (User: {user_id}):\n{chat_history_str}")
        if retrieved_long_term_interactions not in ["No long-term memory available (system not initialized).", "No specific relevant past interactions found in your long-term memory.", "Error retrieving from your long-term memory.", "User ID not provided for memory retrieval."]:
            print(f"[DEBUG] Passing retrieved long-term interactions to agent (User: {user_id}):\n{retrieved_long_term_interactions}")

        agent_input = {
            "input": last_user_message_content, "chat_history": chat_history_str,
            "retrieved_long_term_interactions": retrieved_long_term_interactions
        }

        if stream:
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
                                new_content_to_stream = full_chunk_output; accumulated_final_output = "" 
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
                    print(f"[ERROR] Error during agent stream generation: {e_stream}"); traceback.print_exc()
                    if not is_done_sent: 
                        error_response = {
                            "model": DEFAULT_MODEL, "created_at": datetime.now(timezone.utc).isoformat(),
                            "error": f"Error during agent execution: {str(e_stream)}", "done": True 
                        }
                        yield f'{json.dumps(error_response)}\n'
                print(f"[INFO] Ollama-compatible stream to client finished (User: {user_id}).")
            return Response(stream_with_context(generate_ollama_compatible_stream()), mimetype='application/x-ndjson')
        
        else: # Non-streaming response
            response_payload = agent_executor.invoke(agent_input) 
            agent_final_answer = response_payload.get("output", "Sorry, I could not process your request effectively.")
            assistant_metadata = {"user_id": user_id, "conversation_id": conversation_id}
            add_interaction_to_db(user_id, conversation_id, current_turn_id + 1, "assistant", agent_final_answer, assistant_metadata)
            assistant_doc_id = f"{user_id}_{conversation_id}_turn_{current_turn_id + 1}_assistant"
            add_interaction_to_vector_store(assistant_doc_id, f"Assistant responds: {agent_final_answer}", user_id, assistant_metadata)
            ollama_response = {
                "model": DEFAULT_MODEL, "created_at": datetime.now(timezone.utc).isoformat(),
                "message": { "role": "assistant", "content": agent_final_answer }, "done": True,
            }
            return jsonify(ollama_response)

    except Exception as e:
        print(f"[ERROR] Unhandled exception in LangChain chat_proxy: {type(e).__name__} - {e}"); traceback.print_exc()
        return jsonify({"error": f"Internal Server Error in Agent: {str(e)}", "model": DEFAULT_MODEL, "done": True }), 500

@app.route("/api/tags", methods=["GET"])
def list_ollama_models():
    try:
        import requests 
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        response.raise_for_status()
        return jsonify(response.json()), response.status_code
    except Exception as e: 
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

import json
import os

MEMORY_PATH = "data/memory_store.json"

def load_memory():
    if not os.path.exists(MEMORY_PATH):
        return {}
    with open(MEMORY_PATH, "r") as f:
        return json.load(f)

def save_memory(mem):
    with open(MEMORY_PATH, "w") as f:
        json.dump(mem, f, indent=2)

def enrich_prompt_with_url(url: str) -> str:
    mem = load_memory()
    return mem.get(url, f"No stored memory for {url}")

def store_url_memory(url: str, content: str):
    mem = load_memory()
    mem[url] = content
    save_memory(mem)

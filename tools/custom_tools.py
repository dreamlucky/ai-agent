# tools/custom_tools.py
from langchain.tools import Tool
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup # For the new fetch_webpage tool

# Tool Function: DuckDuckGo Search
def duckduckgo_search_func(query: str, max_results: int = 3) -> str:
    """
    Performs a DuckDuckGo search and returns a formatted string of results.
    Input should be a search query.
    """
    print(f"[INFO in duckduckgo_search_func] Searching for: {query} (max_results: {max_results})")
    try:
        with DDGS() as ddgs:
            # ddgs.text() returns a list of dictionaries.
            # Each dict: {'title': '...', 'href': '...', 'body': '...'}
            results = list(ddgs.text(query, max_results=max_results))
        
        if not results:
            print("[INFO in duckduckgo_search_func] No results found.")
            return "No search results found."

        # Format results into a string
        output_parts = [f"Search results for '{query}':"]
        for i, result in enumerate(results):
            title = result.get('title', 'N/A')
            snippet = result.get('body', 'N/A')
            # href = result.get('href', '#') # You could include the URL if desired
            output_parts.append(f"{i+1}. Title: {title}\n   Snippet: {snippet}")
        
        formatted_results = "\n".join(output_parts)
        print(f"[INFO in duckduckgo_search_func] Formatted results:\n{formatted_results}")
        return formatted_results
    except Exception as e:
        print(f"[ERROR in duckduckgo_search_func] Search failed: {str(e)}")
        return f"DuckDuckGo Search failed due to an error: {str(e)}"

# Tool Function: Fetch Webpage Content
def fetch_webpage_content_func(url: str) -> str:
    """
    Fetches and returns the textual content of a given URL.
    Input must be a valid URL (e.g., http://example.com).
    """
    print(f"[INFO in fetch_webpage_content_func] Fetching content from URL: {url}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        } # Be a good web citizen
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        
        # Use BeautifulSoup to parse HTML and extract text
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements, as they don't contribute to readable content
        for script_or_style in soup(["script", "style", "header", "footer", "nav", "aside"]):
            script_or_style.decompose()
            
        # Get text using .get_text() with a separator to space out text from different tags
        text = soup.get_text(separator='\n', strip=True)
        
        # Basic cleaning: remove excessive newlines
        lines = (line.strip() for line in text.splitlines())
        text_content = '\n'.join(line for line in lines if line)
        
        if not text_content:
            print(f"[INFO in fetch_webpage_content_func] Could not extract meaningful text from {url}.")
            return f"Could not extract meaningful text content from {url}."
        
        # Optional: Truncate if too long for the LLM context
        # Average token is ~4 chars. 16k context / 4 = 4000 tokens. Let's aim for less.
        max_chars = 12000 # Roughly 3000 tokens. Adjust based on your model's context window and typical usage.
        if len(text_content) > max_chars:
            text_content = text_content[:max_chars] + f"\n... (content truncated due to length. Original length: {len(text_content)} chars)"
            print(f"[INFO in fetch_webpage_content_func] Content truncated for {url}.")
            
        print(f"[INFO in fetch_webpage_content_func] Successfully fetched and processed content from {url}.")
        return f"Content from {url}:\n{text_content}"
    except requests.exceptions.HTTPError as e:
        print(f"[ERROR in fetch_webpage_content_func] HTTP error fetching URL {url}: {str(e)}")
        return f"Error fetching URL {url}: HTTP {e.response.status_code}. Content might be inaccessible or protected."
    except requests.exceptions.RequestException as e:
        print(f"[ERROR in fetch_webpage_content_func] Request error fetching URL {url}: {str(e)}")
        return f"Error fetching URL {url}: {str(e)}. Please ensure it's a valid and accessible URL."
    except Exception as e:
        print(f"[ERROR in fetch_webpage_content_func] General error processing URL {url}: {str(e)}")
        return f"Error processing URL {url}: {str(e)}"

# --- Define LangChain Tools ---

# Tool 1: DuckDuckGo Search
search_tool = Tool(
    name="DuckDuckGoSearch",
    func=duckduckgo_search_func,
    description="Useful for when you need to answer questions about current events, facts, or general knowledge. Input should be a search query string."
)

# Tool 2: Fetch Webpage Content
fetch_webpage_tool = Tool(
    name="FetchWebpageContent",
    func=fetch_webpage_content_func,
    description="Useful for when you need to get the content of a specific webpage. Input must be a complete and valid URL (e.g., 'http://example.com')."
)

# --- List of all tools for the agent ---
# The agent_executor in proxy.py will use this list.
agent_tools = [search_tool, fetch_webpage_tool]

# You can add more tools here by:
# 1. Defining their Python function (e.g., a Wikipedia search function).
# 2. Wrapping the function in a `Tool` object with a name and description.
# 3. Adding the new `Tool` object to the `agent_tools` list.
# Example:
# def wikipedia_search_func(query: str) -> str: ...
# wikipedia_tool = Tool(name="WikipediaSearch", func=wikipedia_search_func, description="...")
# agent_tools.append(wikipedia_tool)

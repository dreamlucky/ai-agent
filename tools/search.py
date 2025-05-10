# tools/search.py
from duckduckgo_search import DDGS # Ensure this import is correct for your version

def run_duckduckgo(query: str, max_results: int = 3) -> str:
    """
    Performs a DuckDuckGo search and returns a formatted string of results.
    """
    print(f"[INFO in tools.search.run_duckduckgo] Searching for: {query} (max_results: {max_results})")
    try:
        with DDGS() as ddgs:
            # Note: ddgs.text() returns a list of dictionaries.
            # Each dictionary typically has 'title', 'href', 'body'.
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            print("[INFO in tools.search.run_duckduckgo] No results found.")
            return "No search results found."

        # Format results into a string
        output_parts = [f"Search results for '{query}':"]
        for i, result in enumerate(results):
            title = result.get('title', 'N/A')
            snippet = result.get('body', 'N/A')
            # href = result.get('href', '#') # You could include the URL if desired
            output_parts.append(f"{i+1}. Title: {title}\n   Snippet: {snippet}")

        formatted_results = "\n".join(output_parts)
        print(f"[INFO in tools.search.run_duckduckgo] Formatted results:\n{formatted_results}")
        return formatted_results

    except Exception as e:
        print(f"[ERROR in tools.search.run_duckduckgo] Search failed: {str(e)}")
        return f"DuckDuckGo Search failed due to an error: {str(e)}"

# Example usage (for testing this script directly if you run: python tools/search.py)
# if __name__ == '__main__':
#     print("Testing run_duckduckgo...")
#     test_query = "latest AI advancements"
#     search_output = run_duckduckgo(test_query)
#     print(f"\n--- Test Output for '{test_query}' ---\n{search_output}")
#     print("\n--- Testing with no results (example) ---")
#     # This query is unlikely to return nothing, but demonstrates the path
#     search_output_empty = run_duckduckgo("asdflkjhgasdflkjhgasdflkjhg") 
#     print(search_output_empty)
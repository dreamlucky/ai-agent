import requests

def run_duckduckgo(query):
    try:
        url = f"https://lite.duckduckgo.com/lite/?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            # Crude extraction of first few results (since no API key is needed)
            return f"Search successful. View results at: {url}"
        else:
            return f"DuckDuckGo returned error code {resp.status_code}"
    except Exception as e:
        return f"Search failed: {str(e)}"

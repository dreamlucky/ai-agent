import requests
from bs4 import BeautifulSoup

def scrape_url(url: str, max_chars: int = 3000):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text(separator=' ', strip=True)[:max_chars]
    except Exception as e:
        return f"Error: {e}"

from langchain.tools import DuckDuckGoSearchRun

def get_duckduckgo_tool():
    return DuckDuckGoSearchRun()

def run_langchain_duckduckgo(query: str) -> str:
    tool = get_duckduckgo_tool()
    return tool.run(query)

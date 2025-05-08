import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get config from env
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://192.168.1.8:11434")
DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Placeholder for your agent's entry point
def run_ai_agent():
    logger.info("Starting AI agent")
    logger.debug("Using OLLAMA_BASE_URL: %s", OLLAMA_BASE_URL)

    # TODO: Load models, prompts, memory, etc.
    # Example placeholder
    try:
        logger.info("Connecting to Ollama model backend...")
        # Simulate a connection check or API call
        # response = requests.get(f"{OLLAMA_BASE_URL}/status")
        # logger.debug("Ollama response: %s", response.json())
    except Exception as e:
        logger.error("Failed to connect to Ollama", exc_info=True)
        return

    logger.info("AI agent is now running.")


if __name__ == "__main__":
    run_ai_agent()

import os
import time
import logging

from openai import OpenAI
from openai import APIError, RateLimitError, APIConnectionError

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Default values
MODEL_NAME = "gpt-4-turbo"
MAX_RETRIES = 2
TEMPERATURE = 0.0
RETRY_DELAY = 1  # seconds

# Set up OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    error_msg = "OPENAI_API_KEY environment variable not set"
    logger.error(error_msg)
    raise ValueError(error_msg)

# Initialize the client
client = OpenAI(api_key=api_key)


def call_llm(
    prompt: str, *, model: str = MODEL_NAME, temperature: float = TEMPERATURE
) -> str:
    messages = [
        {"role":"system","content":"You are a JSON‐only response engine.  Always respond with valid JSON only, no markdown, no prose."},
        {"role": "user",   "content": prompt}
    ]
    
    retries = 0
    while retries <= MAX_RETRIES:
        try:
            logger.debug(f"Calling OpenAI API with model {model}")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
            
        except RateLimitError as e:
            retries += 1
            if retries <= MAX_RETRIES:
                wait_time = RETRY_DELAY * retries
                logger.warning(f"Rate limit hit, retrying in {wait_time}s... ({retries}/{MAX_RETRIES})")
                time.sleep(wait_time)
            else:
                logger.error("Rate limit exceeded and max retries reached")
                raise
                
        except APIError as e:
            retries += 1
            if retries <= MAX_RETRIES:
                wait_time = RETRY_DELAY * retries
                logger.warning(f"API error, retrying in {wait_time}s... ({retries}/{MAX_RETRIES})")
                time.sleep(wait_time)
            else:
                logger.error(f"API error after {MAX_RETRIES} retries: {str(e)}")
                raise
                
        except APIConnectionError as e:
            retries += 1
            if retries <= MAX_RETRIES:
                wait_time = RETRY_DELAY * retries
                logger.warning(f"Connection error, retrying in {wait_time}s... ({retries}/{MAX_RETRIES})")
                time.sleep(wait_time)
            else:
                logger.error(f"Connection error after {MAX_RETRIES} retries: {str(e)}")
                raise
                
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI API: {str(e)}")
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    test_prompt = input("Enter a prompt (or press Enter for default): ").strip()
    if not test_prompt:
        test_prompt = "Tell me a joke about programming."
    
    try:
        response = call_llm(test_prompt)
        print("\nResponse from LLM:")
        print("-" * 40)
        print(response)
        print("-" * 40)
    except Exception as e:
        print(f"Error: {e}")
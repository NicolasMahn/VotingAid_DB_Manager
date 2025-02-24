import openai
import tiktoken
from scrt import OPENAI_KEY

WHITE = "\033[97m"
BLUE = "\033[34m"
GREEN = "\033[32m"
ORANGE = "\033[38;5;208m"
PINK = "\033[38;5;205m"
RESET = "\033[0m"

MAX_TOKENS = {
    "default": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "o1": 200000,
    "o1-mini": 128000,
    "o3-mini": 200000
}
DEFAULT_MODEL = "gpt-4o-mini"

def count_context_length(prompt: str, model: str = "default") -> int:
    if model not in MAX_TOKENS or model == "default":
        model = DEFAULT_MODEL
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(prompt))

def model_max_context_length(model: str) -> int:
    if model in MAX_TOKENS:
        return MAX_TOKENS[model]
    return MAX_TOKENS["default"]

def is_context_too_long(prompt: str, model: str = "default") -> bool:
    return count_context_length(prompt, model) > model_max_context_length(model)

def basic_prompt(prompt: str, role :str = "You are a helpful assistant.", temperature: float = 0.2,
                 model: str ="default", debug: bool = False) -> str:
    if model not in MAX_TOKENS or model == "default":
        model = DEFAULT_MODEL
    if is_context_too_long(prompt, model):
        raise ValueError("Prompt exceeds the maximum token limit.")

    if debug:
        print(f"{PINK}ROLE:\n{role}{RESET}")
        print(f"{BLUE}PROMPT:\n{prompt}{RESET}")

    # OpenAI is the default for now
    response = _basic_prompt_openai(prompt, role, temperature, model)


    if debug:
        print(f"{GREEN}RESPONSE:\n{response}{RESET}")
        print(f"---")
    return response


def _basic_prompt_openai(prompt: str, role: str, temperature: float, model: str) -> str:
    openai.api_key = OPENAI_KEY

    response_text = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": role},
            {
                "role": "user",
                "content": prompt,
                "temperature": temperature
            }
        ]
    )
    return response_text.choices[0].message.content
# utils/config_loader.py

"""
This is a simple utility module for loading and accessing model configurations from a central
JSON file (`model_config.json`).

By centralizing configuration, it becomes easier to manage different models (e.g., for different
LLM providers) and their associated parameters like API keys, endpoints, and model names.

The configuration is loaded once when the module is first imported and stored in a global
variable to ensure efficient access throughout the application's lifecycle.
"""
import json
import os

# Define the directory where configuration files are stored.
CONFIG_DIR = "configs"


def resolve_env_placeholders(value):
    """Resolve ${ENV_VAR} placeholders in loaded configuration values."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1], "")
    if isinstance(value, dict):
        return {k: resolve_env_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_env_placeholders(item) for item in value]
    return value


def load_model_configs():
    """
    Loads all model configurations from the 'model_config.json' file.

    Returns:
        A list of dictionaries, where each dictionary represents a model configuration.
        Returns an empty list if the file is not found or contains invalid JSON.
    """
    file_path = os.path.join(CONFIG_DIR, "model_config.json")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return resolve_env_placeholders(json.load(f))
    except FileNotFoundError:
        print(f"Error: Config file model_config.json not found in {CONFIG_DIR}.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing model_config.json: {e}")
        return []

# Load configurations into a global variable upon module import for application-wide access.
GLOBAL_MODEL_CONFIGS = load_model_configs()

def get_model_config(config_name: str):
    """
    Retrieves a specific model's configuration dictionary by its 'config_name'.

    Args:
        config_name: The name of the configuration to retrieve (e.g., "deepseek_chat").

    Returns:
        The configuration dictionary if found, otherwise None.
    """
    for config in GLOBAL_MODEL_CONFIGS:
        if config.get("config_name") == config_name:
            return config
    return None

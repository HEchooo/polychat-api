from pydantic.v1 import BaseSettings
import logging
from typing import Dict, Optional


class LLMSettings(BaseSettings):
    """
    Multi-provider LLM settings with automatic model routing
    """
    
    # Multi-provider configurations
    # OpenAI
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_KEY: str = ""
    
    # Alibaba Cloud (Qwen)
    ALIBABA_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ALIBABA_API_KEY: str = ""
    
    # Google Cloud (Gemini)
    GCP_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta"
    GCP_API_KEY: str = ""
    
    # Local/Default provider
    LOCAL_BASE_URL: str = "http://127.0.0.1:11434/v1"
    LOCAL_API_KEY: str = "ollama"
    
    # General settings
    LLM_MAX_STEP: int = 25
    MAX_CHAT_HISTORY = 10

    class Config(object):
        env_file = ".env"
    
    def get_provider_config(self, model: str) -> Dict[str, str]:
        """
        Get provider configuration based on model name
        Returns dict with 'base_url' and 'api_key'
        """
        model_lower = model.lower()
        
        # OpenAI models
        if any(prefix in model_lower for prefix in ['gpt-', 'gpt4', 'gpt3', 'text-davinci', 'text-curie']):
            if self.OPENAI_KEY:
                logging.info(f"Using OpenAI provider for model: {model}")
                return {"base_url": self.OPENAI_BASE_URL, "api_key": self.OPENAI_KEY}
        
        # Qwen models (Alibaba Cloud)
        if any(prefix in model_lower for prefix in ['qwen', 'qwen2', 'qwen-']):
            if self.ALIBABA_API_KEY:
                logging.info(f"Using Alibaba Cloud provider for model: {model}")
                return {"base_url": self.ALIBABA_BASE_URL, "api_key": self.ALIBABA_API_KEY}
        
        # Gemini models (Google Cloud)
        if any(prefix in model_lower for prefix in ['gemini', 'gemini-pro', 'gemini-']):
            if self.GCP_API_KEY:
                logging.info(f"Using Google Cloud provider for model: {model}")
                return {"base_url": self.GCP_BASE_URL, "api_key": self.GCP_API_KEY}
        
        # Default to local provider
        logging.info(f"Using local provider for model: {model}")
        return {"base_url": self.LOCAL_BASE_URL, "api_key": self.LOCAL_API_KEY}


class ToolSettings(BaseSettings):
    """
    tool settings
    """

    TOOL_WORKER_NUM: int = 10
    TOOL_WORKER_EXECUTION_TIMEOUT: int = 180

    BING_SEARCH_URL: str = "https://api.bing.microsoft.com/v7.0/search"
    BING_SUBSCRIPTION_KEY: str = "xxxx"
    WEB_SEARCH_NUM_RESULTS: int = 5

    R2R_BASE_URL: str = "http://127.0.0.1:8000"
    R2R_USERNAME: str = None
    R2R_PASSWORD: str = None
    R2R_SEARCH_LIMIT: int = 10
    SPECIAL_STREAM_TOOLS: list[str] = ["product_recommendation_api"]
    SPECIAL_NORMAL_TOOLS: list[str] = ["combine_search_v2"]
    
    FILTER_TAGS: list[str] = ["NC0002"]

    class Config(object):
        env_file = ".env"


tool_settings = ToolSettings()
llm_settings = LLMSettings()

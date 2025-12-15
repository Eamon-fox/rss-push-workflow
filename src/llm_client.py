"""LLM Client - Unified interface for different providers."""

import httpx
from typing import Literal

Provider = Literal["zhipu", "deepseek", "openai"]


class LLMClient:
    """Unified LLM client supporting multiple providers."""

    # Provider configurations
    PROVIDERS = {
        "zhipu": {
            "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
            "default_model": "glm-4-flash",
            "headers": {
                "User-Agent": "Cline-VSCode-Extension",
                "HTTP-Referer": "https://cline.bot",
                "X-Title": "Cline",
                "X-Cline-Version": "3.42.0",
            },
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "default_model": "deepseek-chat",
            "headers": {},
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "default_model": "gpt-4o-mini",
            "headers": {},
        },
    }

    def __init__(self, provider: Provider, api_key: str, model: str | None = None):
        """
        Initialize LLM client.

        Args:
            provider: One of "zhipu", "deepseek", "openai"
            api_key: API key for the provider
            model: Model name (uses provider default if not specified)
        """
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")

        config = self.PROVIDERS[provider]
        self.base_url = config["base_url"]
        self.model = model or config["default_model"]
        self.api_key = api_key

        # Build headers
        self.headers = {
            **config["headers"],
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, prompt: str, max_tokens: int = 2000) -> str:
        """
        Send chat completion request.

        Args:
            prompt: User message
            max_tokens: Maximum tokens in response

        Returns:
            Response content string
        """
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }

        response = httpx.post(
            url,
            headers=self.headers,
            json=payload,
            timeout=60,
        )

        if response.status_code != 200:
            raise Exception(f"API error {response.status_code}: {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"]


# Quick test function
def test_provider(provider: Provider, api_key: str) -> None:
    """Test a provider with a simple prompt."""
    print(f"Testing {provider}...")
    print(f"  Base URL: {LLMClient.PROVIDERS[provider]['base_url']}")
    print(f"  Model: {LLMClient.PROVIDERS[provider]['default_model']}")

    client = LLMClient(provider, api_key)
    response = client.chat("用一句话介绍你自己")

    print(f"  Response: {response}")
    print("  ✓ Success!")


if __name__ == "__main__":
    import os
    import sys

    if len(sys.argv) < 2:
        print("Usage: python llm_client.py <provider>")
        print("  provider: zhipu, deepseek, openai")
        print("")
        print("Set API key via environment variable:")
        print("  ZHIPU_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY")
        sys.exit(1)

    provider = sys.argv[1]
    key_map = {
        "zhipu": "ZHIPU_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "openai": "OPENAI_API_KEY",
    }

    api_key = os.getenv(key_map.get(provider, ""))
    if not api_key:
        print(f"Error: {key_map.get(provider)} not set")
        sys.exit(1)

    test_provider(provider, api_key)

"""LLM Client - Zhipu GLM-4-Flash with Cline headers."""

import httpx

# Zhipu Cline endpoint (triggers monthly subscription)
BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
MODEL = "glm-4-flash"

HEADERS = {
    "User-Agent": "Cline-VSCode-Extension",
    "HTTP-Referer": "https://cline.bot",
    "X-Title": "Cline",
    "X-Cline-Version": "3.42.0",
}


def chat(prompt: str, api_key: str, max_tokens: int = 2000) -> str:
    """
    Send chat request to Zhipu GLM-4-Flash.

    Args:
        prompt: User message
        api_key: Zhipu API key
        max_tokens: Maximum tokens in response

    Returns:
        Response content string
    """
    url = f"{BASE_URL}/chat/completions"

    headers = {
        **HEADERS,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }

    response = httpx.post(url, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text}")

    data = response.json()
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    import os

    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        print("Error: ZHIPU_API_KEY not set")
        exit(1)

    print("Testing Zhipu GLM-4-Flash...")
    result = chat("用一句话介绍你自己", api_key)
    print(f"Response: {result}")

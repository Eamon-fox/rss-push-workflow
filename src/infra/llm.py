"""LLM Client - Zhipu GLM-4-Flash."""

import httpx

BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
MODEL = "glm-4-flash"
API_KEY = "b10ed78a78a4477c86ad67d17be00b3b.qEcEy7Hcq8v8XEYk"

HEADERS = {
    "User-Agent": "Cline-VSCode-Extension",
    "HTTP-Referer": "https://cline.bot",
    "X-Title": "Cline",
    "X-Cline-Version": "3.42.0",
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def chat(prompt: str, max_tokens: int = 2000) -> str:
    """
    Send chat request to Zhipu GLM-4-Flash.

    Args:
        prompt: User message
        max_tokens: Maximum tokens in response

    Returns:
        Response content string
    """
    response = httpx.post(
        f"{BASE_URL}/chat/completions",
        headers=HEADERS,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        },
        timeout=120,
    )

    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text}")

    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print(chat("用一句话介绍你自己"))

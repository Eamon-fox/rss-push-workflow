"""LLM client supporting Ollama (local), Zhipu (cloud), SiliconFlow (cloud), and MiMo (cloud) providers."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import yaml
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path("config/settings.yaml")

# Ollama defaults
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = "qwen3:8b"

# Zhipu defaults
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
ZHIPU_SUPPORTED_MODELS = {"glm-4.6", "glm-4.5-air"}
ZHIPU_DEFAULT_MODEL = "glm-4.6"
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY", "")

# SiliconFlow defaults
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_DEFAULT_MODEL = "Qwen/Qwen3-8B"
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")

# MiMo defaults
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO_DEFAULT_MODEL = "mimo-v2-flash"
MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "")


def _build_zhipu_headers() -> dict[str, str]:
    """Build headers for Zhipu API."""
    if not ZHIPU_API_KEY:
        raise RuntimeError("Missing ZHIPU_API_KEY for Zhipu client")
    return {
        "User-Agent": "Cline-VSCode-Extension",
        "HTTP-Referer": "https://cline.bot",
        "X-Title": "Cline",
        "X-Cline-Version": "3.42.0",
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }


def _build_siliconflow_headers() -> dict[str, str]:
    """Build headers for SiliconFlow API."""
    if not SILICONFLOW_API_KEY:
        raise RuntimeError("Missing SILICONFLOW_API_KEY for SiliconFlow client")
    return {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }


def _build_mimo_headers() -> dict[str, str]:
    """Build headers for MiMo API."""
    if not MIMO_API_KEY:
        raise RuntimeError("Missing MIMO_API_KEY for MiMo client")
    return {
        "api-key": MIMO_API_KEY,
        "Content-Type": "application/json",
    }


@lru_cache(maxsize=1)
def _llm_config() -> dict[str, Any]:
    """Load LLM configuration from settings.yaml."""
    config: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        try:
            config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        except Exception:
            config = {}

    settings = config.get("llm") or {}
    provider = settings.get("provider", "ollama")

    # Ollama config
    ollama_cfg = settings.get("ollama") or {}
    ollama_base_url = ollama_cfg.get("base_url", OLLAMA_BASE_URL)
    ollama_model = ollama_cfg.get("model", OLLAMA_DEFAULT_MODEL)

    # Zhipu config
    zhipu_cfg = settings.get("zhipu") or {}
    zhipu_model = zhipu_cfg.get("model", ZHIPU_DEFAULT_MODEL)
    if zhipu_model not in ZHIPU_SUPPORTED_MODELS:
        zhipu_model = ZHIPU_DEFAULT_MODEL
    zhipu_allow_thinking = bool(zhipu_cfg.get("allow_thinking", False))

    # SiliconFlow config
    siliconflow_cfg = settings.get("siliconflow") or {}
    siliconflow_model = siliconflow_cfg.get("model", SILICONFLOW_DEFAULT_MODEL)
    siliconflow_enable_thinking = bool(siliconflow_cfg.get("enable_thinking", False))

    # MiMo config
    mimo_cfg = settings.get("mimo") or {}
    mimo_model = mimo_cfg.get("model", MIMO_DEFAULT_MODEL)
    mimo_enable_thinking = bool(mimo_cfg.get("enable_thinking", False))

    # Concurrency settings
    ollama_concurrency = int(ollama_cfg.get("concurrency", 2))
    zhipu_concurrency = int(zhipu_cfg.get("concurrency", 5))
    siliconflow_concurrency = int(siliconflow_cfg.get("concurrency", 5))
    mimo_concurrency = int(mimo_cfg.get("concurrency", 5))

    return {
        "provider": provider,
        "ollama_base_url": ollama_base_url,
        "ollama_model": ollama_model,
        "ollama_concurrency": ollama_concurrency,
        "zhipu_model": zhipu_model,
        "zhipu_allow_thinking": zhipu_allow_thinking,
        "zhipu_concurrency": zhipu_concurrency,
        "siliconflow_model": siliconflow_model,
        "siliconflow_enable_thinking": siliconflow_enable_thinking,
        "siliconflow_concurrency": siliconflow_concurrency,
        "mimo_model": mimo_model,
        "mimo_enable_thinking": mimo_enable_thinking,
        "mimo_concurrency": mimo_concurrency,
    }


def get_concurrency() -> int:
    """Get concurrency setting for current provider."""
    cfg = _llm_config()
    if cfg["provider"] == "zhipu":
        return cfg["zhipu_concurrency"]
    elif cfg["provider"] == "siliconflow":
        return cfg["siliconflow_concurrency"]
    elif cfg["provider"] == "mimo":
        return cfg["mimo_concurrency"]
    return cfg["ollama_concurrency"]


def _chat_ollama(
    prompt: str,
    max_tokens: int,
    temperature: float,
    model: str | None = None,
) -> str:
    """Send chat request to Ollama."""
    cfg = _llm_config()
    base_url = cfg["ollama_base_url"]
    effective_model = model or cfg["ollama_model"]

    payload = {
        "model": effective_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    response = httpx.post(
        f"{base_url}/api/chat",
        json=payload,
        timeout=180,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Ollama API error {response.status_code}: {response.text}")

    data = response.json()
    message = data.get("message", {})
    content = (message.get("content") or "").strip()

    # qwen3 thinking mode: remove <think>...</think> tags
    content = _strip_think_tags(content)

    return content


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks if they somehow appear in content.

    Note: Ollama normally separates 'content' and 'thinking' fields,
    but this is a safety net in case they get mixed.
    """
    if not text or "<think>" not in text:
        return text

    import re
    # Remove <think>...</think> blocks (closed)
    text = re.sub(r"<think>[\s\S]*?</think>", "", text)
    # Remove unclosed <think> to end (truncated)
    text = re.sub(r"<think>[\s\S]*$", "", text)
    return text.strip()


def _chat_siliconflow(
    prompt: str,
    max_tokens: int,
    temperature: float,
    model: str | None = None,
    enable_thinking: bool | None = None,
) -> str:
    """Send chat request to SiliconFlow."""
    cfg = _llm_config()
    effective_model = model or cfg["siliconflow_model"]
    thinking_enabled = enable_thinking if enable_thinking is not None else cfg["siliconflow_enable_thinking"]

    payload: dict[str, Any] = {
        "model": effective_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.85,  # 略微提高采样范围
        "frequency_penalty": 0.3,  # 学术摘要允许适度重复术语
        "enable_thinking": thinking_enabled,
    }

    response = httpx.post(
        f"{SILICONFLOW_BASE_URL}/chat/completions",
        headers=_build_siliconflow_headers(),
        json=payload,
        timeout=180,
    )
    if response.status_code != 200:
        raise RuntimeError(f"SiliconFlow API error {response.status_code}: {response.text}")

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message", {})
    content = (message.get("content") or "").strip()
    return content


def _chat_zhipu(
    prompt: str,
    max_tokens: int,
    temperature: float,
    model: str | None = None,
    allow_thinking: bool | None = None,
) -> str:
    """Send chat request to Zhipu."""
    cfg = _llm_config()
    effective_model = model if model in ZHIPU_SUPPORTED_MODELS else cfg["zhipu_model"]
    thinking_enabled = allow_thinking if allow_thinking is not None else cfg["zhipu_allow_thinking"]

    payload: dict[str, Any] = {
        "model": effective_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "thinking": {"type": "enabled"} if thinking_enabled else {"type": "disabled"},
    }

    response = httpx.post(
        f"{ZHIPU_BASE_URL}/chat/completions",
        headers=_build_zhipu_headers(),
        json=payload,
        timeout=120,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Zhipu API error {response.status_code}: {response.text}")

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message", {})
    content = (message.get("content") or "").strip()
    reasoning = message.get("reasoning_content")
    if not content and isinstance(reasoning, str):
        content = reasoning.strip()
    return content


def _chat_mimo(
    prompt: str,
    max_tokens: int,
    temperature: float,
    model: str | None = None,
    enable_thinking: bool | None = None,
) -> str:
    """Send chat request to MiMo."""
    cfg = _llm_config()
    effective_model = model or cfg["mimo_model"]
    thinking_enabled = enable_thinking if enable_thinking is not None else cfg["mimo_enable_thinking"]

    payload: dict[str, Any] = {
        "model": effective_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_completion_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "thinking": {"type": "enabled"} if thinking_enabled else {"type": "disabled"},
    }

    response = httpx.post(
        f"{MIMO_BASE_URL}/chat/completions",
        headers=_build_mimo_headers(),
        json=payload,
        timeout=180,
    )
    if response.status_code != 200:
        raise RuntimeError(f"MiMo API error {response.status_code}: {response.text}")

    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message", {})
    content = (message.get("content") or "").strip()
    return content


def chat(
    prompt: str,
    max_tokens: int = 30000,
    temperature: float = 0.4,  # 降低以获得更稳定的摘要输出
    *,
    model: str | None = None,
    allow_thinking: bool | None = None,
) -> str:
    """
    Send a chat completion request and return assistant text content.
    Automatically routes to configured provider (ollama, zhipu, siliconflow, or mimo).
    """
    cfg = _llm_config()
    provider = cfg["provider"]

    if provider == "zhipu":
        return _chat_zhipu(prompt, max_tokens, temperature, model, allow_thinking)
    elif provider == "siliconflow":
        return _chat_siliconflow(prompt, max_tokens, temperature, model, allow_thinking)
    elif provider == "mimo":
        return _chat_mimo(prompt, max_tokens, temperature, model, allow_thinking)
    else:
        # Default to ollama
        return _chat_ollama(prompt, max_tokens, temperature, model)


if __name__ == "__main__":
    print(f"Provider: {_llm_config()['provider']}")
    print(chat("请用一句话介绍你自己。"))

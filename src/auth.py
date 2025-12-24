"""微信登录 + JWT 认证模块."""

import time
from pathlib import Path
from typing import Optional

import httpx
import jwt
import yaml

from src.core import load_json, save_json, now_iso
from src.user_config import get_or_create_config

# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
USERS_FILE = DATA_DIR / "users.json"
CONFIG_FILE = Path("config/settings.yaml")

# JWT 配置
JWT_SECRET = "scholarpipe-secret-key-change-in-production"
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 7


def _load_config() -> dict:
    """加载微信配置"""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config.get("wechat", {})
    return {}


def _load_users() -> dict:
    """加载用户数据"""
    return load_json(USERS_FILE, default={})


def _save_users(users: dict):
    """保存用户数据"""
    save_json(users, USERS_FILE)


# ─────────────────────────────────────────────────────────────
# 微信登录
# ─────────────────────────────────────────────────────────────

async def wx_code2session(code: str) -> dict:
    """
    调用微信 code2session 接口

    Args:
        code: 小程序 wx.login() 获取的 code

    Returns:
        {"openid": "...", "session_key": "..."} 或错误信息
    """
    config = _load_config()
    appid = config.get("appid")
    secret = config.get("secret")

    if not appid or not secret:
        raise ValueError("微信 appid/secret 未配置")

    url = "https://api.weixin.qq.com/sns/jscode2session"
    params = {
        "appid": appid,
        "secret": secret,
        "js_code": code,
        "grant_type": "authorization_code",
    }

    async with httpx.AsyncClient() as client:
        resp = await client.get(url, params=params, timeout=10)
        data = resp.json()

    if "errcode" in data and data["errcode"] != 0:
        raise ValueError(f"微信登录失败: {data.get('errmsg', 'Unknown error')}")

    return data


async def wx_login(code: str) -> dict:
    """
    微信登录流程

    Args:
        code: 小程序 wx.login() 获取的 code

    Returns:
        {"token": "...", "openid": "...", "is_new_user": bool}
    """
    # 1. 调用微信接口换取 openid
    wx_data = await wx_code2session(code)
    openid = wx_data["openid"]

    # 2. 查找或创建用户
    users = _load_users()
    is_new_user = openid not in users

    if is_new_user:
        users[openid] = {
            "created_at": now_iso(),
            "last_login": now_iso(),
            "login_count": 1,
        }
        # 创建默认用户配置
        get_or_create_config(openid)
    else:
        users[openid]["last_login"] = now_iso()
        users[openid]["login_count"] = users[openid].get("login_count", 0) + 1

    _save_users(users)

    # 3. 生成 JWT token
    token = generate_token(openid)

    return {
        "token": token,
        "openid": openid,
        "is_new_user": is_new_user,
    }


# ─────────────────────────────────────────────────────────────
# JWT Token
# ─────────────────────────────────────────────────────────────

def generate_token(openid: str) -> str:
    """
    生成 JWT token

    Args:
        openid: 用户的微信 openid

    Returns:
        JWT token 字符串
    """
    payload = {
        "openid": openid,
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXPIRE_DAYS * 24 * 3600,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> Optional[str]:
    """
    验证 JWT token

    Args:
        token: JWT token 字符串

    Returns:
        openid 或 None (验证失败)
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload.get("openid")
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def get_user_info(openid: str) -> Optional[dict]:
    """
    获取用户信息

    Args:
        openid: 用户的微信 openid

    Returns:
        用户信息字典或 None
    """
    users = _load_users()
    return users.get(openid)

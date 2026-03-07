"""Tests for app.core.security — JWT creation, password hashing."""

from datetime import timedelta

import jwt

from app.core.security import create_access_token, get_password_hash, verify_password
from app.core.config import settings


def test_create_token_contains_sub():
    token = create_access_token("user-123")
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    assert payload["sub"] == "user-123"
    assert "exp" in payload


def test_create_token_custom_expiry():
    t_short = create_access_token("u", expires_delta=timedelta(minutes=1))
    t_long = create_access_token("u", expires_delta=timedelta(hours=24))
    p_short = jwt.decode(t_short, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    p_long = jwt.decode(t_long, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    assert p_long["exp"] > p_short["exp"]


def test_password_hash_and_verify():
    hashed = get_password_hash("s3cret!")
    assert hashed.startswith("$2b$")
    assert verify_password("s3cret!", hashed) is True
    assert verify_password("wrong", hashed) is False

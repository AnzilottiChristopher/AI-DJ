"""
Authentication router - register, login, me, stats.
Uses JWT tokens and bcrypt password hashing.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr

import bcrypt
import jwt

from database import get_connection

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SECRET_KEY = os.getenv("AIDJ_JWT_SECRET", "aidj-dev-secret-key-change-in-production-env")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 72

router = APIRouter(prefix="/api/auth", tags=["auth"])
security = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: str


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------
def create_access_token(user_id: int, username: str) -> str:
    payload = {
        "sub": str(user_id),
        "username": username,
        "exp": datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ---------------------------------------------------------------------------
# Dependency - get current user from Authorization header
# ---------------------------------------------------------------------------
def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> dict:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    payload = decode_token(credentials.credentials)
    return {"id": int(payload["sub"]), "username": payload["username"]}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.post("/register")
async def register(req: RegisterRequest):
    """Create a new user account."""
    if len(req.username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    password_hash = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt()).decode()

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (req.username, req.email, password_hash),
        )
        conn.commit()
        user_id = cursor.lastrowid
    except Exception as e:
        conn.close()
        if "UNIQUE constraint" in str(e):
            if "username" in str(e):
                raise HTTPException(status_code=409, detail="Username already taken")
            raise HTTPException(status_code=409, detail="Email already registered")
        raise HTTPException(status_code=500, detail="Registration failed")
    finally:
        conn.close()

    token = create_access_token(user_id, req.username)
    return {
        "token": token,
        "user": {
            "id": user_id,
            "username": req.username,
            "email": req.email,
        },
    }


@router.post("/login")
async def login(req: LoginRequest):
    """Authenticate and return a JWT token."""
    conn = get_connection()
    row = conn.execute(
        "SELECT id, username, email, password_hash FROM users WHERE username = ?",
        (req.username,),
    ).fetchone()
    conn.close()

    if row is None:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    if not bcrypt.checkpw(req.password.encode(), row["password_hash"].encode()):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_access_token(row["id"], row["username"])
    return {
        "token": token,
        "user": {
            "id": row["id"],
            "username": row["username"],
            "email": row["email"],
        },
    }


@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    """Return the current user's profile."""
    conn = get_connection()
    row = conn.execute(
        "SELECT id, username, email, created_at FROM users WHERE id = ?",
        (user["id"],),
    ).fetchone()
    conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "id": row["id"],
        "username": row["username"],
        "email": row["email"],
        "created_at": row["created_at"],
    }


@router.get("/stats")
async def get_stats(user: dict = Depends(get_current_user)):
    """Return activity stats for the current user."""
    conn = get_connection()

    counts = conn.execute(
        """
        SELECT event_type, COUNT(*) as count
        FROM activity
        WHERE user_id = ?
        GROUP BY event_type
        """,
        (user["id"],),
    ).fetchall()

    stats = {row["event_type"]: row["count"] for row in counts}
    conn.close()

    return {
        "songs_played": stats.get("song_played", 0),
        "songs_uploaded": stats.get("song_uploaded", 0),
        "prompts_sent": stats.get("prompt_sent", 0),
        "playlists_generated": stats.get("playlist_generated", 0),
    }


@router.post("/activity")
async def log_activity(event: dict, user: dict = Depends(get_current_user)):
    """Log a user activity event."""
    event_type = event.get("event_type")
    detail = event.get("detail")

    if not event_type:
        raise HTTPException(status_code=400, detail="event_type is required")

    conn = get_connection()
    conn.execute(
        "INSERT INTO activity (user_id, event_type, detail) VALUES (?, ?, ?)",
        (user["id"], event_type, detail),
    )
    conn.commit()
    conn.close()

    return {"ok": True}

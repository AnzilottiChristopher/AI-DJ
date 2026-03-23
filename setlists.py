"""
Setlist CRUD router — create, read, update, delete setlists.
Each setlist is an ordered list of songs with optional transition preferences.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth import get_current_user
from database import get_connection

router = APIRouter(prefix="/api/setlists", tags=["setlists"])


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class SetlistItemCreate(BaseModel):
    song_key: str
    song_title: str
    song_artist: str = ""
    transition_exit_segment: str | None = None
    transition_entry_segment: str | None = None


class SetlistCreate(BaseModel):
    name: str
    items: list[SetlistItemCreate] = []


class SetlistUpdate(BaseModel):
    name: str | None = None
    items: list[SetlistItemCreate] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fetch_items(conn, setlist_id: int) -> list[dict]:
    rows = conn.execute(
        "SELECT song_key, song_title, song_artist, transition_exit_segment, transition_entry_segment "
        "FROM setlist_items WHERE setlist_id = ? ORDER BY position",
        (setlist_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _insert_items(conn, setlist_id: int, items: list[SetlistItemCreate]):
    for i, item in enumerate(items):
        conn.execute(
            "INSERT INTO setlist_items (setlist_id, position, song_key, song_title, song_artist, "
            "transition_exit_segment, transition_entry_segment) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (setlist_id, i, item.song_key, item.song_title, item.song_artist,
             item.transition_exit_segment, item.transition_entry_segment),
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@router.get("")
async def list_setlists(user: dict = Depends(get_current_user)):
    """List all setlists for the current user."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT s.id, s.name, s.created_at, s.updated_at, "
        "(SELECT COUNT(*) FROM setlist_items si WHERE si.setlist_id = s.id) AS item_count "
        "FROM setlists s WHERE s.user_id = ? ORDER BY s.updated_at DESC",
        (user["id"],),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.post("", status_code=201)
async def create_setlist(req: SetlistCreate, user: dict = Depends(get_current_user)):
    """Create a new setlist with optional items."""
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Setlist name is required")

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO setlists (user_id, name) VALUES (?, ?)",
            (user["id"], req.name.strip()),
        )
        setlist_id = cursor.lastrowid
        _insert_items(conn, setlist_id, req.items)
        conn.commit()
    except Exception as e:
        conn.close()
        if "UNIQUE constraint" in str(e):
            raise HTTPException(status_code=409, detail="A setlist with that name already exists")
        raise HTTPException(status_code=500, detail="Failed to create setlist")
    finally:
        conn.close()

    return {"id": setlist_id, "name": req.name.strip(), "item_count": len(req.items)}


@router.get("/{setlist_id}")
async def get_setlist(setlist_id: int, user: dict = Depends(get_current_user)):
    """Get a single setlist with all items."""
    conn = get_connection()
    row = conn.execute(
        "SELECT id, name, created_at, updated_at FROM setlists WHERE id = ? AND user_id = ?",
        (setlist_id, user["id"]),
    ).fetchone()

    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="Setlist not found")

    items = _fetch_items(conn, setlist_id)
    conn.close()

    return {**dict(row), "items": items}


@router.put("/{setlist_id}")
async def update_setlist(setlist_id: int, req: SetlistUpdate, user: dict = Depends(get_current_user)):
    """Update a setlist's name and/or items."""
    conn = get_connection()
    row = conn.execute(
        "SELECT id FROM setlists WHERE id = ? AND user_id = ?",
        (setlist_id, user["id"]),
    ).fetchone()

    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="Setlist not found")

    try:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        if req.name is not None:
            conn.execute(
                "UPDATE setlists SET name = ?, updated_at = ? WHERE id = ?",
                (req.name.strip(), now, setlist_id),
            )

        if req.items is not None:
            conn.execute("DELETE FROM setlist_items WHERE setlist_id = ?", (setlist_id,))
            _insert_items(conn, setlist_id, req.items)
            conn.execute(
                "UPDATE setlists SET updated_at = ? WHERE id = ?",
                (now, setlist_id),
            )

        conn.commit()
    except Exception as e:
        conn.close()
        if "UNIQUE constraint" in str(e):
            raise HTTPException(status_code=409, detail="A setlist with that name already exists")
        raise HTTPException(status_code=500, detail="Failed to update setlist")
    finally:
        conn.close()

    return {"ok": True}


@router.delete("/{setlist_id}")
async def delete_setlist(setlist_id: int, user: dict = Depends(get_current_user)):
    """Delete a setlist and all its items."""
    conn = get_connection()
    result = conn.execute(
        "DELETE FROM setlists WHERE id = ? AND user_id = ?",
        (setlist_id, user["id"]),
    )
    conn.commit()
    deleted = result.rowcount
    conn.close()

    if deleted == 0:
        raise HTTPException(status_code=404, detail="Setlist not found")

    return {"ok": True}

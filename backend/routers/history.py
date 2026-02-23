"""
GET /api/history?limit=20
Returns the last N predictions stored in the SQLite database.
"""

from fastapi import APIRouter, Query
from backend.db import fetch_history

router = APIRouter()


@router.get("/history")
def history(limit: int = Query(default=20, ge=1, le=100)):
    """
    Retrieve prediction history.
    Returns a list of past predictions ordered by most recent first.
    """
    rows = fetch_history(limit=limit)
    return {"count": len(rows), "predictions": rows}

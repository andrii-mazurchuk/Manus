from fastapi import APIRouter

router = APIRouter(prefix="/api/sequences", tags=["sequences"])


@router.get("/status")
async def sequences_status():
    """Placeholder — sequence recognition is not yet implemented."""
    return {
        "available": False,
        "note": "Sequence recognition not yet implemented. "
                "Sequences are temporal patterns of static gesture tokens.",
    }

"""Analysis creation and history.

``POST /analyses`` follows REST rather than the RPC-style ``/analyze``: an
analysis is a resource, so creating one and listing them share a single path.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, File, Query, UploadFile, status

from app.api.deps import AnalysisServiceDep, SettingsDep
from app.core.errors import PayloadTooLargeError
from app.schemas.analysis import AnalysisRead
from app.schemas.common import ErrorResponse, Page

router = APIRouter(prefix="/analyses", tags=["analyses"])

_CHUNK_SIZE = 64 * 1024

_ERROR_RESPONSES: dict[int | str, dict] = {
    413: {"model": ErrorResponse, "description": "File exceeds the size limit"},
    415: {"model": ErrorResponse, "description": "Unsupported file type"},
    422: {"model": ErrorResponse, "description": "File is not a readable image"},
    501: {"model": ErrorResponse, "description": "Analysis engine not available"},
}


async def _read_bounded(upload: UploadFile, max_bytes: int) -> bytes:
    """Buffer an upload, refusing to read more than ``max_bytes``.

    Chunked so an oversized file is rejected without being held in memory in full.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(_CHUNK_SIZE):
        total += len(chunk)
        if total > max_bytes:
            raise PayloadTooLargeError(
                f"The uploaded file exceeds the maximum of {max_bytes} bytes.",
                details={"max_bytes": max_bytes},
            )
        chunks.append(chunk)
    return b"".join(chunks)


@router.post(
    "",
    response_model=AnalysisRead,
    status_code=status.HTTP_201_CREATED,
    summary="Analyze an image",
    description=(
        "Uploads an image, analyzes its visual quality and stores the result.\n\n"
        "**Not yet available.** The upload is fully validated, but no analysis "
        "model is loaded in this build, so a validated request returns 501."
    ),
    responses=_ERROR_RESPONSES,
)
async def create_analysis(
    service: AnalysisServiceDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="The image to analyze.")],
) -> AnalysisRead:
    data = await _read_bounded(file, settings.max_upload_bytes)
    return await service.create_analysis(
        data,
        filename=file.filename or "upload",
        declared_media_type=file.content_type,
    )


@router.get(
    "",
    response_model=Page[AnalysisRead],
    summary="List previous analyses",
)
async def list_analyses(
    service: AnalysisServiceDep,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> Page[AnalysisRead]:
    items, total = await service.list_analyses(limit=limit, offset=offset)
    return Page(items=items, total=total, limit=limit, offset=offset)


@router.get(
    "/{analysis_id}",
    response_model=AnalysisRead,
    summary="Retrieve one analysis",
    responses={404: {"model": ErrorResponse, "description": "No such analysis"}},
)
async def get_analysis(analysis_id: uuid.UUID, service: AnalysisServiceDep) -> AnalysisRead:
    return await service.get_analysis(analysis_id)

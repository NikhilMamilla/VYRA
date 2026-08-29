"""Analysis creation and history.

``POST /analyses`` follows REST rather than the RPC-style ``/analyze``: an
analysis is a resource, so creating one and listing them share a single path.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, File, Query, UploadFile, status

from app.api.deps import AnalysisServiceDep, SettingsDep
from app.core.errors import FeatureNotAvailableError, PayloadTooLargeError, VyraError
from app.schemas.analysis import (
    AnalysisRead,
    BatchAnalysisItem,
    BatchAnalysisResponse,
    BatchItemError,
)
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
        "Returns the persisted analysis (`201`). If the deployment has no "
        "inference bundle loaded (`MODEL_PATH` unset), the upload is still "
        "validated but analysis is unavailable and the request returns `501`."
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


@router.post(
    "/batch",
    response_model=BatchAnalysisResponse,
    summary="Analyze up to N images in one request",
    description=(
        "Uploads several images as `multipart/form-data` (repeat the `files` "
        "part). Each image is validated, analysed and persisted independently: "
        "the response is always `200` and reports per-image success or failure "
        "in `items`. Returns `413` only if the number of files exceeds "
        "`MAX_BATCH_SIZE`, or `501` if no analysis model is loaded."
    ),
    responses={
        413: {"model": ErrorResponse, "description": "Too many files in one request"},
        501: {"model": ErrorResponse, "description": "Analysis engine not available"},
    },
)
async def create_analyses_batch(
    service: AnalysisServiceDep,
    settings: SettingsDep,
    files: Annotated[list[UploadFile], File(description="The images to analyze.")],
) -> BatchAnalysisResponse:
    if not service.analyzer_available:
        raise FeatureNotAvailableError(
            "Image analysis is not available in this build. No analysis model is loaded."
        )
    if len(files) > settings.max_batch_size:
        raise PayloadTooLargeError(
            f"A batch may contain at most {settings.max_batch_size} images.",
            details={"max_batch_size": settings.max_batch_size, "received": len(files)},
        )

    items: list[BatchAnalysisItem] = []
    for upload in files:
        name = upload.filename or "upload"
        try:
            data = await _read_bounded(upload, settings.max_upload_bytes)
            analysis = await service.create_analysis(
                data, filename=name, declared_media_type=upload.content_type
            )
            items.append(BatchAnalysisItem(filename=name, ok=True, analysis=analysis))
        except VyraError as exc:
            items.append(
                BatchAnalysisItem(
                    filename=name,
                    ok=False,
                    error=BatchItemError(code=exc.code, message=exc.message),
                )
            )

    succeeded = sum(1 for item in items if item.ok)
    return BatchAnalysisResponse(
        total=len(items),
        succeeded=succeeded,
        failed=len(items) - succeeded,
        items=items,
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

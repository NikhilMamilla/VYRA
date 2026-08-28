"""Partial download of a specific set of images from a large remote ZIP.

The VizWiz ``val.zip`` is 3.5 GB; we only need ~2-3k images for the experiment.
The host supports HTTP range requests, so ``remotezip`` lets us pull just the
entries we want. Downloads are parallelised and cached, so a re-run is free.
"""

from __future__ import annotations

import concurrent.futures
import threading
from pathlib import Path

import remotezip

# (connect, read) timeout for every range request. Without this a single stalled
# socket makes the whole thread-pool join hang forever (observed on the last
# handful of members of a 2.5k fetch against VizWiz's 11 GB train.zip).
_HTTP_TIMEOUT = (10, 30)
# Wall-clock budget for the whole fetch after the cache check. Any member still
# outstanding when this elapses is abandoned; callers already tolerate a missing
# image (recorded as load_status="missing").
_OVERALL_BUDGET_S = 900


def fetch_from_remote_zip(
    zip_url: str,
    members: list[str],
    dest_dir: Path,
    *,
    workers: int = 12,
    on_progress=None,
) -> dict[str, Path]:
    """Download ``members`` (paths inside the zip) into ``dest_dir``.

    Returns ``{member: local_path}`` for every member successfully fetched.
    Already-cached files are skipped. Failures (including timeouts and the
    overall budget being exceeded) are logged and omitted.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}
    todo: list[str] = []
    for member in members:
        local = dest_dir / Path(member).name
        if local.is_file() and local.stat().st_size > 0:
            results[member] = local
        else:
            todo.append(member)

    if not todo:
        return results

    local_zip = threading.local()
    done = 0

    def _client(fresh: bool = False) -> remotezip.RemoteZip:
        if fresh or not hasattr(local_zip, "z"):
            local_zip.z = remotezip.RemoteZip(zip_url, timeout=_HTTP_TIMEOUT)
        return local_zip.z

    def _one(member: str) -> tuple[str, Path | None]:
        for attempt in (0, 1):
            try:
                data = _client(fresh=attempt == 1).read(member)
                break
            except Exception:  # noqa: BLE001 - retry once, then record and continue
                if attempt == 1:
                    return member, None
        local = dest_dir / Path(member).name
        tmp = local.with_suffix(local.suffix + ".part")
        tmp.write_bytes(data)
        tmp.replace(local)
        return member, local

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(_one, m): m for m in todo}
    try:
        for fut in concurrent.futures.as_completed(futures, timeout=_OVERALL_BUDGET_S):
            member, local = fut.result()
            done += 1
            if on_progress and done % 100 == 0:
                on_progress(done, len(todo))
            if local is not None:
                results[member] = local
    except concurrent.futures.TimeoutError:
        pass
    finally:
        for fut in futures:
            fut.cancel()
        pool.shutdown(wait=False, cancel_futures=True)

    if on_progress:
        on_progress(done, len(todo))
    return results


def list_remote_zip(zip_url: str) -> list[str]:
    with remotezip.RemoteZip(zip_url) as z:
        return [n for n in z.namelist() if not n.endswith("/")]

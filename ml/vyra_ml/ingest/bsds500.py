"""BSDS500 as a clean-image source.

The Berkeley Segmentation Dataset (500 natural images, ~72 MB) is used purely as
a supply of diverse, real, reasonably clean photographs to apply synthetic
degradations to. Its segmentation annotations are ignored. It is small enough to
download once inside a 48h assessment and is a long-standing, freely available
computer-vision dataset.

Licensing note: BSDS500 is distributed by UC Berkeley for research and
educational use. We redistribute nothing -- the archive is downloaded at build
time into the git-ignored ``data/raw`` directory.
"""

from __future__ import annotations

import hashlib
import tarfile
from collections.abc import Iterator
from pathlib import Path

import requests

from vyra_ml.ingest.base import SourceAdapter, SourceImage

# The BIDS mirror on GitHub carries the original 500 BSDS images (the Berkeley
# eecs host is frequently unreachable). Same images, same research/educational
# licensing; the segmentation annotations are ignored.
_ARCHIVE_URL = "https://codeload.github.com/BIDS/BSDS500/tar.gz/refs/heads/master"
_ARCHIVE_NAME = "bsds500-master.tar.gz"
# Path fragment every image shares, regardless of the archive's top-level dir.
_IMAGE_MARKER = "BSDS500/data/images/"
_EXTRACTED_DIR = "images"


class BSDS500Adapter(SourceAdapter):
    name = "bsds500"

    @property
    def _image_root(self) -> Path:
        return self.raw_dir / _EXTRACTED_DIR

    def prepare(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        if self._image_root.is_dir() and any(self._image_root.rglob("*.jpg")):
            return

        archive = self.raw_dir / _ARCHIVE_NAME
        if not archive.is_file():
            self._download(archive)
        self._extract(archive)

        if not any(self._image_root.rglob("*.jpg")):
            raise RuntimeError(
                "BSDS500 extracted but no images were found. Inspect "
                f"{self.raw_dir}. As a fallback set dataset.source: skimage in the config."
            )

    def _download(self, dest: Path) -> None:
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with requests.get(_ARCHIVE_URL, stream=True, timeout=60) as response:
                response.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Could not download BSDS500 from {_ARCHIVE_URL}: {exc}. "
                "Check connectivity, or set dataset.source: skimage in the config."
            ) from exc
        tmp.replace(dest)

    def _extract(self, archive: Path) -> None:
        dest_root = self._image_root.resolve()
        dest_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                if not (
                    member.isfile()
                    and _IMAGE_MARKER in member.name
                    and member.name.endswith(".jpg")
                ):
                    continue
                # Flatten to images/<split>/<name>.jpg and guard traversal.
                tail = member.name.split(_IMAGE_MARKER, 1)[1]
                target = (dest_root / tail).resolve()
                if not str(target).startswith(str(dest_root)):
                    raise RuntimeError(f"Unsafe path in archive: {member.name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                src = tar.extractfile(member)
                if src is not None:
                    target.write_bytes(src.read())

    def iter_originals(self, limit: int | None = None) -> Iterator[SourceImage]:
        # Deterministic order: sort by a hash of the filename so the capped
        # subset is a stable, content-spread sample rather than "all of train/".
        paths = sorted(
            self._image_root.rglob("*.jpg"),
            key=lambda p: hashlib.blake2b(p.name.encode(), digest_size=8).digest(),
        )
        if limit is not None:
            paths = paths[:limit]
        for path in paths:
            yield SourceImage(
                source_id=f"bsds500/{path.stem}",
                source_dataset="bsds500",
                path=path,
            )

from __future__ import annotations

import gzip
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO


def _native_io_path(path: Path) -> str:
    """Return an OS path that also supports long absolute paths on Windows."""
    absolute = os.path.abspath(os.fspath(Path(path)))
    if os.name != "nt" or absolute.startswith("\\\\?\\"):
        return absolute
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute[2:]
    return "\\\\?\\" + absolute


def _is_file(path: Path) -> bool:
    return os.path.isfile(_native_io_path(path))


def gzip_sibling(path: Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".gz")


def resolve_text_artifact(path: Path) -> Path:
    path = Path(path)
    if _is_file(path):
        return path
    compressed = gzip_sibling(path)
    return compressed if _is_file(compressed) else path


def artifact_exists(path: Path) -> bool:
    return _is_file(resolve_text_artifact(Path(path)))


@contextmanager
def open_text_artifact(
    path: Path,
    mode: str = "r",
    *,
    encoding: str = "utf-8",
    newline: str | None = None,
) -> Iterator[TextIO]:
    resolved = resolve_text_artifact(Path(path))
    if resolved.suffix == ".gz":
        handle = gzip.open(
            _native_io_path(resolved),
            mode if "t" in mode else mode + "t",
            encoding=encoding,
            newline=newline,
        )
    else:
        handle = open(
            _native_io_path(resolved),
            mode,
            encoding=encoding,
            newline=newline,
        )
    try:
        yield handle
    finally:
        handle.close()


def read_json_artifact(path: Path):
    with open_text_artifact(Path(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_binary_stream(handle) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def compact_to_gzip(
    path: Path,
    *,
    remove_source: bool = True,
    compresslevel: int = 6,
) -> dict:
    source = Path(path)
    destination = gzip_sibling(source)
    if not _is_file(source):
        if _is_file(destination):
            with gzip.open(_native_io_path(destination), "rb") as handle:
                digest, uncompressed_size = _sha256_binary_stream(handle)
            return {
                "status": "already_compacted",
                "source_path": str(source),
                "compressed_path": str(destination),
                "uncompressed_size": int(uncompressed_size),
                "compressed_size": int(os.stat(_native_io_path(destination)).st_size),
                "uncompressed_sha256": digest,
            }
        raise FileNotFoundError(str(source))

    with open(_native_io_path(source), "rb") as handle:
        source_digest, source_size = _sha256_binary_stream(handle)

    if _is_file(destination):
        with gzip.open(_native_io_path(destination), "rb") as handle:
            existing_digest, existing_size = _sha256_binary_stream(handle)
        if existing_digest != source_digest or existing_size != source_size:
            raise RuntimeError(
                f"Existing compressed artifact does not match source: {destination}"
            )
    else:
        temp_path = destination.with_name(
            f".{destination.name}.{os.getpid()}.tmp"
        )
        try:
            with open(_native_io_path(source), "rb") as input_handle, open(
                _native_io_path(temp_path), "wb"
            ) as raw_output:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    compresslevel=int(compresslevel),
                    fileobj=raw_output,
                    mtime=0,
                ) as output_handle:
                    for chunk in iter(
                        lambda: input_handle.read(8 * 1024 * 1024), b""
                    ):
                        output_handle.write(chunk)
            with gzip.open(_native_io_path(temp_path), "rb") as handle:
                verified_digest, verified_size = _sha256_binary_stream(handle)
            if verified_digest != source_digest or verified_size != source_size:
                raise RuntimeError(
                    f"Compressed artifact verification failed: {source}"
                )
            os.replace(_native_io_path(temp_path), _native_io_path(destination))
        finally:
            if os.path.exists(_native_io_path(temp_path)):
                os.unlink(_native_io_path(temp_path))

    if remove_source:
        os.unlink(_native_io_path(source))
    return {
        "status": "compacted",
        "source_path": str(source),
        "compressed_path": str(destination),
        "uncompressed_size": int(source_size),
        "compressed_size": int(os.stat(_native_io_path(destination)).st_size),
        "uncompressed_sha256": source_digest,
    }

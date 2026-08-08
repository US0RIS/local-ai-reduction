"""Minimal LARC v0.1 container prototype."""
from __future__ import annotations
import json, struct
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"LARC\x00\x01\x00\x00"
HEADER = struct.Struct("<8sQ")

@dataclass
class Chunk:
    name: str
    codec: str
    data: bytes
    metadata: dict

def write_larc(path: str | Path, chunks: list[Chunk], metadata: dict | None = None) -> None:
    metadata = dict(metadata or {})
    manifest = {"metadata": metadata, "chunks": []}
    offset = 0
    for chunk in chunks:
        manifest["chunks"].append({"name": chunk.name, "codec": chunk.codec, "offset": offset, "length": len(chunk.data), "metadata": chunk.metadata})
        offset += len(chunk.data)
    payload = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
    with open(path, "wb") as f:
        f.write(HEADER.pack(MAGIC, len(payload))); f.write(payload)
        for chunk in chunks: f.write(chunk.data)

def read_manifest(path: str | Path) -> dict:
    with open(path, "rb") as f:
        magic, n = HEADER.unpack(f.read(HEADER.size))
        if magic != MAGIC: raise ValueError("not a LARC v0.1 file")
        return json.loads(f.read(n))

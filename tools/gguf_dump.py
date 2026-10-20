#!/usr/bin/env python3
"""Write GGUF header-card JSON by piping an SWR1 witness to `gguf-card accept`."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import struct
import subprocess
import sys
from pathlib import Path

PIN = "0.19.0"
ROOT = Path(__file__).resolve().parent.parent

WIDTH = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 4,
    5: 4,
    6: 4,
    7: 1,
    10: 8,
    11: 8,
    12: 8,
}


def u32(value: int) -> bytes:
    return struct.pack("<I", value)


def u64(value: int) -> bytes:
    return struct.pack("<Q", value)


FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def fnv1a64(chunks: list[bytes]) -> int:
    digest = FNV_OFFSET
    for chunk in chunks:
        for byte in chunk:
            digest ^= byte
            digest = (digest * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return digest


def require_pin() -> None:
    try:
        version = importlib.metadata.version("gguf")
    except importlib.metadata.PackageNotFoundError:
        print("gguf is not installed", file=sys.stderr)
        raise SystemExit(2) from None
    if version != PIN:
        print(f"gguf {version} != {PIN}", file=sys.stderr)
        raise SystemExit(2)


def build_witness(reader) -> bytes:
    meta = []
    for name, field in reader.fields.items():
        if name.startswith("GGUF."):
            continue
        meta.append(encode_meta(field))
    tensors = [encode_tensor(tensor) for tensor in reader.tensors]
    version = int(reader.fields["GGUF.version"].parts[0][0])
    out = bytearray()
    out += b"SWR1"
    out += u64(version)
    out += u64(int(reader.alignment))
    out += u64(int(reader.data_offset))
    out += u64(len(meta))
    for item in meta:
        out += item
    out += u64(len(tensors))
    for item in tensors:
        out += item
    return bytes(out)


def encode_meta(field) -> bytes:
    key = field.parts[1].tobytes()
    type_id = int(field.types[0])
    return u64(len(key)) + key + u32(type_id) + encode_value(field, type_id)


def encode_value(field, type_id: int) -> bytes:
    if type_id == 9:
        return encode_array(field)
    raw = field.parts[field.data[0]].tobytes()
    if type_id == 8:
        return u64(len(raw)) + raw
    width = WIDTH.get(type_id)
    if width is None or len(raw) != width:
        raise SystemExit(f"{field.name}: type {type_id} payload is {len(raw)} bytes")
    return raw


def encode_array(field) -> bytes:
    elem = int(field.parts[3][0])
    length = int(field.parts[4][0])
    if len(field.types) > 1 and int(field.types[-1]) != elem:
        raise SystemExit(f"{field.name}: element type {elem} != types[-1]")
    if elem == 9:
        raise SystemExit(f"{field.name}: nested array")
    if len(field.data) != length:
        raise SystemExit(f"{field.name}: sent {len(field.data)} elements, declared {length}")
    out = bytearray()
    out += u32(elem)
    out += u64(length)
    out += u64(length)
    hashed: list[bytes] = []
    for index in field.data:
        raw = field.parts[index].tobytes()
        if elem == 8:
            piece = u64(len(raw)) + raw
            out += piece
            hashed.append(piece)
            continue
        width = WIDTH.get(elem)
        if width is None or len(raw) != width:
            raise SystemExit(f"{field.name}: element type {elem} payload is {len(raw)} bytes")
        out += raw
        hashed.append(raw)
    out += u64(fnv1a64(hashed))
    return bytes(out)


def encode_tensor(tensor) -> bytes:
    field = tensor.field
    name = field.parts[1].tobytes()
    dims = field.parts[3]
    type_id = int(field.parts[4][0])
    offset = int(field.parts[5][0])
    out = bytearray()
    out += u64(len(name))
    out += name
    out += u64(len(dims))
    for dim in dims:
        out += u64(int(dim))
    out += u32(type_id)
    out += u64(offset)
    return bytes(out)


def accept(witness: bytes) -> bytes:
    proc = subprocess.run(
        ["cargo", "run", "-q", "-p", "gguf", "--bin", "gguf-card", "--", "accept"],
        input=witness,
        stdout=subprocess.PIPE,
        cwd=ROOT,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.returncode or 1)
    return proc.stdout


def write_out(path: Path, payload: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    require_pin()
    import gguf

    reader = gguf.GGUFReader(args.model, "r")
    payload = accept(build_witness(reader))
    if args.out is None:
        sys.stdout.buffer.write(payload)
        return
    write_out(args.out, payload)


if __name__ == "__main__":
    main()

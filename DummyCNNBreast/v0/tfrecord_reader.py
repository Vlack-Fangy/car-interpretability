"""
Minimal pure-Python TFRecord reader for the skooch/ddsm-mammography shards.
No TensorFlow dependency.

TFRecord frame format (per record):
    uint64  length          (little endian)
    uint32  masked_crc32c(length_bytes)      <- not verified here
    bytes   payload[length]                  <- serialized tf.train.Example
    uint32  masked_crc32c(payload)           <- not verified here

We skip CRC verification (needs crc32c, not zlib's IEEE crc32). Corrupt shards
would surface as a protobuf parse failure instead, which is loud enough.
"""
import struct
from typing import Dict, Iterator, Tuple


# ---------- varint / protobuf primitives ----------

def _read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _read_tag(buf: bytes, pos: int) -> Tuple[int, int, int]:
    key, pos = _read_varint(buf, pos)
    return key >> 3, key & 0x07, pos


def _skip_field(buf: bytes, pos: int, wire: int) -> int:
    if wire == 0:
        _, pos = _read_varint(buf, pos)
    elif wire == 1:
        pos += 8
    elif wire == 2:
        n, pos = _read_varint(buf, pos)
        pos += n
    elif wire == 5:
        pos += 4
    else:
        raise ValueError(f"unsupported wire type {wire}")
    return pos


# ---------- tf.train.Example parsing ----------

def _parse_int64_list(buf: bytes) -> list:
    out, pos, n = [], 0, len(buf)
    while pos < n:
        field, wire, pos = _read_tag(buf, pos)
        if field == 1 and wire == 2:          # packed
            ln, pos = _read_varint(buf, pos)
            end = pos + ln
            while pos < end:
                v, pos = _read_varint(buf, pos)
                out.append(v)
        elif field == 1 and wire == 0:        # unpacked
            v, pos = _read_varint(buf, pos)
            out.append(v)
        else:
            pos = _skip_field(buf, pos, wire)
    return out


def _parse_bytes_list(buf: bytes) -> list:
    out, pos, n = [], 0, len(buf)
    while pos < n:
        field, wire, pos = _read_tag(buf, pos)
        if field == 1 and wire == 2:
            ln, pos = _read_varint(buf, pos)
            out.append(buf[pos:pos + ln])
            pos += ln
        else:
            pos = _skip_field(buf, pos, wire)
    return out


def _parse_feature(buf: bytes):
    pos, n = 0, len(buf)
    while pos < n:
        field, wire, pos = _read_tag(buf, pos)
        if wire != 2:
            pos = _skip_field(buf, pos, wire)
            continue
        ln, pos = _read_varint(buf, pos)
        payload = buf[pos:pos + ln]
        pos += ln
        if field == 1:
            return ("bytes", _parse_bytes_list(payload))
        if field == 3:
            return ("int64", _parse_int64_list(payload))
        # field 2 == float_list, unused by this dataset
    return (None, None)


def parse_example(buf: bytes) -> Dict[str, object]:
    """Serialized tf.train.Example -> {feature_name: value}."""
    feats: Dict[str, object] = {}
    pos, n = 0, len(buf)
    while pos < n:
        field, wire, pos = _read_tag(buf, pos)          # Example.features = 1
        if field != 1 or wire != 2:
            pos = _skip_field(buf, pos, wire)
            continue
        ln, pos = _read_varint(buf, pos)
        fbuf, fend = buf[pos:pos + ln], pos + ln
        pos = fend

        p2, n2 = 0, len(fbuf)
        while p2 < n2:                                   # Features.feature = 1 (map)
            f2, w2, p2 = _read_tag(fbuf, p2)
            if f2 != 1 or w2 != 2:
                p2 = _skip_field(fbuf, p2, w2)
                continue
            ln2, p2 = _read_varint(fbuf, p2)
            entry, p2 = fbuf[p2:p2 + ln2], p2 + ln2

            key, val, p3 = None, None, 0
            while p3 < len(entry):                       # MapEntry: key=1, value=2
                f3, w3, p3 = _read_tag(entry, p3)
                if w3 != 2:
                    p3 = _skip_field(entry, p3, w3)
                    continue
                ln3, p3 = _read_varint(entry, p3)
                blob, p3 = entry[p3:p3 + ln3], p3 + ln3
                if f3 == 1:
                    key = blob.decode("utf-8")
                elif f3 == 2:
                    val = _parse_feature(blob)
            if key is not None:
                kind, vals = val
                if kind == "bytes":
                    feats[key] = vals[0] if len(vals) == 1 else vals
                elif kind == "int64":
                    feats[key] = vals[0] if len(vals) == 1 else vals
    return feats


# ---------- record framing ----------

def iter_tfrecord(path: str) -> Iterator[Dict[str, object]]:
    with open(path, "rb") as f:
        while True:
            header = f.read(8)
            if len(header) < 8:
                return
            (length,) = struct.unpack("<Q", header)
            f.read(4)                       # masked crc of length
            payload = f.read(length)
            if len(payload) < length:
                raise EOFError("truncated tfrecord")
            f.read(4)                       # masked crc of payload
            yield parse_example(payload)
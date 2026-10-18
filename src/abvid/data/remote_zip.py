"""Metadata-only reading of remote (optionally split) ZIP archives.

Only the end-of-central-directory records, the central directory and explicitly
allowlisted members are transferred, using HTTP byte ranges. Audio members are
listed from the central directory but their bytes are never requested.
"""
from __future__ import annotations

import re
import struct
import urllib.request
import zlib
from dataclasses import dataclass

EOCD = 0x06054B50
ZIP64_LOCATOR = 0x07064B50
ZIP64_EOCD = 0x06064B50
CENTRAL = 0x02014B50
LOCAL = 0x04034B50
SPLIT = 0x08074B50
AUDIO_SUFFIXES = (".flac", ".wav", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".aif", ".aiff")


class Volumes:
    """Concatenated byte view over split-archive volumes, in disk order.

    ``fetch(index, start, length)`` returns bytes from one volume. Every request
    is counted so a caller can report and cap the bytes actually transferred.
    """

    def __init__(self, sizes, fetch, *, max_bytes):
        if not sizes or any(s <= 0 for s in sizes):
            raise ValueError("Expected positive volume sizes")
        self.sizes, self.fetch, self.max_bytes = list(sizes), fetch, max_bytes
        self.starts = [sum(self.sizes[:i]) for i in range(len(self.sizes))]
        self.transferred, self.requests = [0] * len(self.sizes), []

    def read(self, disk, offset, length):
        if not 0 <= disk < len(self.sizes) or offset < 0 or length < 0:
            raise ValueError("Read outside the archive volumes")
        position, chunks = self.starts[disk] + offset, []
        while length:
            volume = max(i for i, s in enumerate(self.starts) if s <= position)
            local = position - self.starts[volume]
            take = min(length, self.sizes[volume] - local)
            if take <= 0:
                raise ValueError("Read beyond the final volume")
            if sum(self.transferred) + take > self.max_bytes:
                raise RuntimeError("Metadata transfer budget exceeded")
            data = self.fetch(volume, local, take)
            if len(data) != take:
                raise IOError("Short range response")
            self.transferred[volume] += take
            self.requests.append([volume, local, take])
            chunks.append(data)
            position, length = position + take, length - take
        return b"".join(chunks)

    def tail(self, length):
        last = len(self.sizes) - 1
        take = min(length, self.sizes[last])
        return self.read(last, self.sizes[last] - take, take), self.sizes[last] - take


@dataclass(frozen=True)
class Member:
    name: str
    method: int
    flags: int
    crc32: int
    compressed_size: int
    size: int
    disk: int
    header_offset: int

    @property
    def is_audio(self):
        return self.name.lower().endswith(AUDIO_SUFFIXES)


def _zip64_values(extra, wanted):
    """Read ZIP64 extra fields in the order mandated by the specification."""
    position = 0
    while position + 4 <= len(extra):
        tag, size = struct.unpack_from("<HH", extra, position)
        body = extra[position + 4:position + 4 + size]
        if tag == 1:
            values, cursor = {}, 0
            for name, width in wanted:
                fmt = "<Q" if width == 8 else "<I"
                values[name] = struct.unpack_from(fmt, body, cursor)[0]
                cursor += width
            return values
        position += 4 + size
    raise ValueError("ZIP64 field required but absent")


def central_directory(volumes):
    """Return (members, record) using only the archive tail and directory."""
    tail, _ = volumes.tail(65_557)
    at = tail.rfind(struct.pack("<I", EOCD))
    if at < 0:
        raise ValueError("End of central directory not found")
    (_, this_disk, cd_disk, _, entries, cd_size, cd_offset,
     comment) = struct.unpack_from("<IHHHHIIH", tail, at)
    record = {"eocd_disk": this_disk, "zip64": False}
    if 0xFFFF in (this_disk, cd_disk, entries) or 0xFFFFFFFF in (cd_size, cd_offset):
        locator = at - 20
        if locator < 0 or struct.unpack_from("<I", tail, locator)[0] != ZIP64_LOCATOR:
            raise ValueError("ZIP64 locator not found")
        _, z64_disk, z64_offset, total_disks = struct.unpack_from("<IIQI", tail, locator)
        raw = volumes.read(z64_disk, z64_offset, 56)
        if struct.unpack_from("<I", raw)[0] != ZIP64_EOCD:
            raise ValueError("Invalid ZIP64 end record")
        (_, _, _, _, this_disk, cd_disk, _, entries, cd_size,
         cd_offset) = struct.unpack_from("<IQHHIIQQQQ", raw)
        record.update(zip64=True, declared_disks=total_disks)
    if this_disk != len(volumes.sizes) - 1:
        raise ValueError("Volume count disagrees with the archive end record")
    data = volumes.read(cd_disk, cd_offset, cd_size)
    members, cursor = [], 0
    for _ in range(entries):
        if struct.unpack_from("<I", data, cursor)[0] != CENTRAL:
            raise ValueError("Corrupt central directory")
        (_, _, _, flags, method, _, _, crc, csize, usize, nlen, xlen, clen, disk, _, _,
         offset) = struct.unpack_from("<IHHHHHHIIIHHHHHII", data, cursor)
        name = data[cursor + 46:cursor + 46 + nlen].decode("utf-8" if flags & 0x800 else "cp437")
        extra = data[cursor + 46 + nlen:cursor + 46 + nlen + xlen]
        wanted = [(k, w) for k, v, w in [("size", usize, 8), ("compressed_size", csize, 8),
                                       ("offset", offset, 8), ("disk", disk, 4)]
                  if v == (0xFFFF if k == "disk" else 0xFFFFFFFF)]
        if wanted:
            values = _zip64_values(extra, wanted)
            usize, csize = values.get("size", usize), values.get("compressed_size", csize)
            offset, disk = values.get("offset", offset), values.get("disk", disk)
        members.append(Member(name, method, flags, crc, csize, usize, disk, offset))
        cursor += 46 + nlen + xlen + clen
    record.update(entries=entries, central_directory_disk=cd_disk,
                  central_directory_offset=cd_offset, central_directory_size=cd_size,
                  comment_length=comment)
    return members, record


def read_member(volumes, member, *, allow):
    """Read one allowlisted, non-audio member and verify its CRC32."""
    if member.is_audio or not re.fullmatch(allow, member.name):
        raise PermissionError(f"Member is not allowlisted metadata: {member.name}")
    if member.flags & 1:
        raise ValueError("Encrypted members are not supported")
    if member.size > 50_000_000:
        raise ValueError("Metadata member is unexpectedly large")
    header = volumes.read(member.disk, member.header_offset, 30)
    if struct.unpack_from("<I", header)[0] != LOCAL:
        raise ValueError("Invalid local header")
    nlen, xlen = struct.unpack_from("<HH", header, 26)
    start = member.header_offset + 30 + nlen + xlen
    payload = volumes.read(member.disk, start, member.compressed_size)
    if member.method == 0:
        data = payload
    elif member.method == 8:
        data = zlib.decompress(payload, -15)
    else:
        raise ValueError(f"Unsupported compression method {member.method}")
    if len(data) != member.size or zlib.crc32(data) & 0xFFFFFFFF != member.crc32:
        raise ValueError(f"Integrity check failed: {member.name}")
    return data


def http_fetcher(urls, *, timeout=120):
    """Range-request fetcher; rejects servers that ignore the requested range."""
    def fetch(volume, start, length):
        request = urllib.request.Request(urls[volume], headers={
            "Range": f"bytes={start}-{start + length - 1}", "User-Agent": "abvid-metadata-admission"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 206:
                raise IOError("Server did not honour the byte range")
            expected = f"bytes {start}-{start + length - 1}/"
            if not response.headers.get("Content-Range", "").startswith(expected):
                raise IOError("Unexpected Content-Range")
            return response.read()
    return fetch

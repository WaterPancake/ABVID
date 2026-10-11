"""Create-only, hash-verified artifact bundles for the grouped CAST runner.

Callers resolve approved storage roots with ``abvid.paths`` before calling this
module. No interrupted attempt directory is removed, and no existing bundle is
replaced. Symlinks, undeclared files, and incomplete bundles fail closed.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import uuid


MANIFEST = "_bundle.json"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _name(value):
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError("Payload name must be a relative POSIX path")
    path = PurePosixPath(value)
    if (not path.parts or path.is_absolute() or str(path) != value or
            any(part in (".", "..", MANIFEST) for part in path.parts)):
        raise ValueError("Unsafe or reserved payload path")
    return value


def _folder(folder):
    path = Path(folder)
    if ".." in path.parts:
        raise ValueError("Bundle path cannot contain traversal")
    path = path.absolute()
    # Do not resolve first: that would silently accept a symlinked ancestor.
    for ancestor in reversed((path, *path.parents)):
        try:
            info = ancestor.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"Symlink in bundle path: {ancestor}")
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"Bundle ancestor is not a directory: {ancestor}")
    return path


def _inventory(folder):
    """Inspect every entry without opening any file, including the manifest."""
    files, directories = set(), set()
    if not folder.is_dir():
        raise ValueError("Bundle directory is missing or incomplete")
    pending = [folder]
    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                relative = (current / entry.name).relative_to(folder).as_posix()
                if stat.S_ISLNK(info.st_mode):
                    raise ValueError(f"Symlink in bundle inventory: {relative}")
                if stat.S_ISDIR(info.st_mode):
                    directories.add(relative)
                    pending.append(current / entry.name)
                elif stat.S_ISREG(info.st_mode):
                    files.add(relative)
                else:
                    raise ValueError(f"Nonregular bundle entry: {relative}")
    return files, directories


def _read_file(path):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError("Bundle payload must be a regular file")
        return handle.read()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate key in bundle manifest")
        result[key] = value
    return result


def _manifest(metadata, hashes):
    if not isinstance(metadata, dict):
        raise ValueError("Bundle metadata must be a JSON object")
    # Reject values or key coercions that would not survive the JSON round trip.
    decoded = json.loads(_canonical(metadata))
    if decoded != metadata:
        raise ValueError("Bundle metadata must round-trip as canonical JSON")
    body = {"metadata": decoded, "payload_sha256": dict(sorted(hashes.items()))}
    return {**body, "identity_sha256": _sha(_canonical(body))}


def _load(folder):
    folder = _folder(folder)
    files, directories = _inventory(folder)
    if MANIFEST not in files:
        raise ValueError("Bundle manifest is missing; bundle is incomplete")
    raw_manifest = _read_file(folder / MANIFEST)
    manifest = json.loads(raw_manifest, object_pairs_hook=_unique_object)
    if not isinstance(manifest, dict) or set(manifest) != {
            "metadata", "payload_sha256", "identity_sha256"}:
        raise ValueError("Invalid bundle manifest schema")
    hashes = manifest["payload_sha256"]
    if not isinstance(hashes, dict):
        raise ValueError("Invalid bundle payload inventory")
    for name, value in hashes.items():
        _name(name)
        if (not isinstance(value, str) or len(value) != 64 or
                any(character not in "0123456789abcdef" for character in value)):
            raise ValueError("Invalid payload SHA-256")
    expected = _manifest(manifest["metadata"], hashes)
    if manifest != expected or raw_manifest != _canonical(expected) + b"\n":
        raise ValueError("Bundle manifest identity or canonical encoding differs")
    expected_directories = {str(ancestor) for name in hashes for ancestor in
                            PurePosixPath(name).parents if str(ancestor) != "."}
    if files != set(hashes) | {MANIFEST} or directories != expected_directories:
        raise ValueError("Bundle inventory differs from declared payloads")
    payload = {name: _read_file(folder / name) for name in sorted(hashes)}
    if {name: _sha(value) for name, value in payload.items()} != hashes:
        raise ValueError("Bundle payload hash differs")
    return manifest, payload


def verify_bundle(folder):
    """Verify the complete inventory and all hashes, returning its manifest."""
    return _load(folder)[0]


def read_bundle(folder):
    """Return verified metadata and exact payload bytes; never accept extras."""
    manifest, payload = _load(folder)
    return manifest["metadata"], payload


def _durable_write(path, value):
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def _publication_lock(folder):
    # Serialize this helper's publishers. An interrupted lock is deliberately
    # not stolen: a human can inspect it together with the preserved attempt.
    lock = folder.parent / ("." + folder.name + ".bundle-publish.lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.close(descriptor)
        yield
    finally:
        lock.unlink()


def write_bundle(folder, payload, metadata):
    """Stage and atomically publish a bundle, or verify an identical existing one.

    Attempt directories are intentionally retained after any failed write or
    rename. This helper never repairs, removes, or overwrites existing data.
    """
    if not isinstance(payload, dict) or any(not isinstance(value, bytes) for value in payload.values()):
        raise ValueError("Payload must map relative names to bytes")
    for name in payload:
        _name(name)
    # A payload file cannot also be a directory required by another payload.
    if any(str(ancestor) in payload for name in payload
           for ancestor in PurePosixPath(name).parents if str(ancestor) != "."):
        raise ValueError("Payload file and directory names collide")
    manifest = _manifest(metadata, {name: _sha(value) for name, value in payload.items()})
    folder = _folder(folder)
    if folder.exists():
        if verify_bundle(folder) != manifest:
            raise ValueError("Existing bundle differs; immutable bundle cannot be replaced")
        return manifest
    folder.parent.mkdir(parents=True, exist_ok=True)
    _folder(folder)
    with _publication_lock(folder):
        if folder.exists():
            if verify_bundle(folder) != manifest:
                raise ValueError("Existing bundle differs; immutable bundle cannot be replaced")
            return manifest
        attempt = folder.parent / (".attempt-" + uuid.uuid4().hex)
        attempt.mkdir(mode=0o700)
        for name, value in sorted(payload.items()):
            destination = attempt / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            _durable_write(destination, value)
        _durable_write(attempt / MANIFEST, _canonical(manifest) + b"\n")
        if verify_bundle(attempt) != manifest:
            raise ValueError("Staged bundle failed verification")
        _folder(folder)
        if folder.exists():
            raise FileExistsError("Destination appeared during publication; attempt retained")
        os.rename(attempt, folder)
    return manifest

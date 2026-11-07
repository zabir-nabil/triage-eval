import json
import os
import tempfile
from pathlib import Path


def _path(path):
    return Path(os.path.expandvars(str(path))).expanduser()


def read_json(path):
    with _path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def _atomic_write(path, content):
    destination = _path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_json(path, data):
    _atomic_write(
        path,
        json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
    )


def read_jsonl(path):
    source = _path(path)
    records = []
    with source.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source}:{number}: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{source}:{number}: expected an object")
            records.append(record)
    return records


def write_jsonl(path, records):
    content = "".join(
        json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
        for record in records
    )
    _atomic_write(path, content)


def read_records(path):
    source = _path(path)
    if source.suffix.lower() == ".jsonl":
        return read_jsonl(source)
    data = read_json(source)
    if isinstance(data, dict):
        data = data.get("records")
    if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
        raise ValueError(f"{source}: expected an array of records or a records object")
    return data

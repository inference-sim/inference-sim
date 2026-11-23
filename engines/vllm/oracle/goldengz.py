# Original to blis-vllm.
"""Reading and writing the goldens the repository stores gzip-compressed.

A golden of GZIP_MIN bytes or more is committed as <name>.gz, so the repository's module stays under
100 MB (internal/repocheck TestModuleZipIsUnder100MB); a smaller one stays plain. The Go tests read
either form through internal/testgz. Every script that reads a golden names the plain path and
calls load() or read_bytes(), which find either form; every script that writes one calls store(),
or pack() once a runner has written a directory:

    python3 -I oracle/goldengz.py pack <golden dir>...

and python3 -I oracle/goldengz.py unpack <golden dir> <plain copy> gives a script that reads a
golden directory itself (a runner's --check) a decompressed copy, and `cat <file>` prints one
golden.

The runners are not edited to do it themselves: a golden records the sha256 of every oracle file
its run executed (`oracle_inputs`), and check_goldens.py holds it to them, so a change to a runner
would make every golden it produced stale. Standard library only.
"""
import gzip
import io
import json
import os

GZIP_MIN = 16000


def exists(path: str) -> bool:
    return os.path.exists(path) or os.path.exists(path + ".gz")


def read_bytes(path: str) -> bytes:
    """The file at path, or the gunzipped path + ".gz" when only that is committed."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    with gzip.open(path + ".gz", "rb") as f:
        return f.read()


def load(path: str):
    return json.loads(read_bytes(path))


def listdir(d: str) -> list:
    """The names in directory d, with a ".gz" suffix removed, sorted and without repeats."""
    return sorted({n[:-3] if n.endswith(".gz") else n for n in os.listdir(d)})


def store_bytes(path: str, data: bytes) -> None:
    """Write data to path, or to path + ".gz" (mtime 0, so the bytes are reproducible) when it is
    GZIP_MIN bytes or more; the other form, if present, is removed."""
    if len(data) < GZIP_MIN:
        with open(path, "wb") as f:
            f.write(data)
        stale = path + ".gz"
    else:
        buf = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=buf, compresslevel=9, mtime=0) as g:
            g.write(data)
        with open(path + ".gz", "wb") as f:
            f.write(buf.getvalue())
        stale = path
    if os.path.exists(stale):
        os.remove(stale)


def store(path: str, text: str) -> None:
    store_bytes(path, text.encode())


def pack(d: str) -> None:
    """Store every plain .json file of directory d in the form store() chooses."""
    if not os.path.isdir(d):
        return
    for n in sorted(os.listdir(d)):
        if n.endswith(".json"):
            p = os.path.join(d, n)
            with open(p, "rb") as f:
                data = f.read()
            store_bytes(p, data)


def unpack(src: str, dst: str) -> None:
    """Copy directory src to dst with every ".gz" file decompressed, for a script that reads a
    golden directory itself and is not edited (a runner's --check)."""
    os.makedirs(dst, exist_ok=True)
    for n in sorted(os.listdir(src)):
        if os.path.isfile(os.path.join(src, n)):
            with open(os.path.join(dst, n[:-3] if n.endswith(".gz") else n), "wb") as f:
                f.write(read_bytes(os.path.join(src, n[:-3] if n.endswith(".gz") else n)))


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "pack":
        for d in sys.argv[2:]:
            pack(d)
    elif len(sys.argv) == 4 and sys.argv[1] == "unpack":
        unpack(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] == "cat":
        sys.stdout.buffer.write(read_bytes(sys.argv[2]))
    else:
        sys.exit("usage: python3 -I oracle/goldengz.py pack <golden dir>...\n"
                 "       python3 -I oracle/goldengz.py unpack <golden dir> <plain copy>\n"
                 "       python3 -I oracle/goldengz.py cat <golden file>")

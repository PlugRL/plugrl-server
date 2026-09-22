"""Is a safetensors checkpoint complete, or was its write cut short?

The file is 8 bytes of little-endian header length, that many bytes of JSON
header, then the tensor data. The header declares every tensor's byte range,
so the size the file OUGHT to be is computable without reading the payload -
and comparing it against the size on disk says whether the write finished.
"""

import json
import pathlib
import struct
import sys

for path in sys.argv[1:]:
    p = pathlib.Path(path)
    actual = p.stat().st_size
    with p.open("rb") as fh:
        raw = fh.read(8)
        if len(raw) < 8:
            print(f"{p.name}\tTRUNCATED\tfile shorter than the 8-byte length prefix")
            continue
        (hlen,) = struct.unpack("<Q", raw)
        head = fh.read(hlen)
        if len(head) < hlen:
            print(f"{p.name}\tTRUNCATED\theader itself is incomplete")
            continue
        meta = json.loads(head)

    end = 0
    n = 0
    for name, spec in meta.items():
        if name == "__metadata__":
            continue
        n += 1
        end = max(end, spec["data_offsets"][1])

    expected = 8 + hlen + end
    verdict = "COMPLETE" if actual == expected else "TRUNCATED"
    print(
        f"{p.parent.name}\t{verdict}\ttensors={n}\texpected={expected}\t"
        f"actual={actual}\tmissing={expected - actual}"
    )

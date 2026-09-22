"""What dtypes does a checkpoint store, and are they the ones the model holds?

`load_state_dict` copies into existing parameters, so it casts the stored
tensor to the parameter's dtype. That cast is the one remaining place a
strict, correctly-keyed round trip could still lose something: float32 stored
into a bfloat16 parameter drops sixteen bits of mantissa per element.

Read from the safetensors header only - no tensor data is loaded.
"""

import collections
import json
import pathlib
import struct
import sys

for path in sys.argv[1:]:
    p = pathlib.Path(path)
    with p.open("rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        meta = json.loads(fh.read(hlen))

    counts = collections.Counter(
        spec["dtype"] for name, spec in meta.items() if name != "__metadata__"
    )
    print(f"{p.parent.name}\t{dict(counts)}")

# E2: the wire protocol can be implemented from the specification alone

2026-09-09 · WSL2 Ubuntu 22.04 · Python 3.11 · plugrl-server @ 4aaebe0

## In one sentence

Two clients - one in Python with neither numpy nor any PlugRL package, one in
**C++ with no third-party libraries at all** - each drove a real
plugrl-server through 120 complete training exchanges. "An environment client
need not be this codebase" stopped being a claim and became a demonstration.

## The stronger result: the C++ client

`plugrl_client.cpp`, built with g++ 11.4, a 68 KB binary. `ldd` says:

```
linux-vdso.so.1
libstdc++.so.6
libgcc_s.so.1
libc.so.6
libm.so.6
```

**Nothing beyond the C++ standard library and libc.** No msgpack library, no
WebSocket library, no OpenSSL - SHA-1, base64, WebSocket framing and masking,
and the subset of msgpack the protocol needs are all written by hand in that
one file, against the specification.

| Measure | Value |
|---|---|
| Exchanges completed | **120** |
| Wall clock | 22.4s |
| Server `global_step` | **100 / 300** |
| Action returned | `dtype=<f4 shape=[4,1,7]`, real float values |

That is the situation an embedded robot controller is actually in: a
compiler and a TCP socket, and nothing else.

## The second result: a Python client with no numpy and no PlugRL

Done first, as a quick check that the protocol could be implemented from the
specification. The client's entire virtual environment:

```
Package    Version
msgpack    1.2.2
websockets 15.0.1
```

| Measure | Value |
|---|---|
| infer/action/feedback exchanges | **120** |
| Wall clock | 22.6s |
| Server `global_step` | **100 / 300** |
| Server `total_connections` | 1 |
| Server collection_time / learn_time | 11.2s / 10.0s |
| Action returned | `dtype=<f4 shape=[4, 1, 7]`, real float values |
| Uplink per step | two uint8 images (224x224x3 + 112x112x3, ~187 KB) |

**The server did not merely tolerate these messages - it advanced its
training progress and completed a learn phase.**

## What the protocol asks of an implementer

### Entirely standard

* **Transport**: ordinary WebSocket. A hand-written HTTP upgrade request
  (`printf` piped to `nc`) is enough to get `101 Switching Protocols` and
  then the metadata frame. No custom handshake, no authentication, no
  subprotocol negotiation.
* **Payload**: standard msgpack. A general-purpose decoder parses it with no
  extensions.
* **Message flow**: metadata (the server speaks first), then a loop of
  infer, action, feedback. Four message types, four lowercase strings.

### The one numpy-specific convention

Arrays travel as a map with binary keys:

```
{b"__ndarray__": true, b"data": <bin>, b"dtype": "<f4", b"shape": [4, 1, 7]}
```

`dtype` is a numpy typestr - byte order, kind, item size. It is the only
piece of numpy vocabulary in the protocol, it is a documented fixed format,
and both clients parse it by hand in about ten lines.

One detail worth knowing: numpy's `bool_` is one byte per element, and
Python's `array` module has no boolean typecode, so booleans have to be
handled as raw bytes - which is what a C++ client would do anyway.

## What this establishes

1. The transport is ordinary WebSocket, verified twice - once with a
   hand-written HTTP upgrade, once with hand-written framing.
2. The payload is standard msgpack, verified twice - once with a
   general-purpose decoder, once with a hand-written one.
3. The single numpy convention parses in about ten lines, implemented in both
   clients.
4. **Both zero-dependency clients advanced real training progress.**

So the implementation burden on the environment side is a WebSocket client
and a msgpack codec, both of which have mature implementations in every
mainstream language, and can also be written from scratch. A design that
requires the environment side to run a full Python framework process cannot
say that.

## A defect found along the way: the dummy policy's batch inference

`dummy_policy._infer_batch_size` took `next(iter(obs.values()))` and then
`len()`. An observation is `{"images": {...}, "states": {...}, "text": [...]}`,
so what it actually measured was the **number of camera names**, not the
batch size. The first run sent `"images": {}`, the server computed batch=0 and
returned an empty action of `shape=[4, 0, 7]` - **without raising**.

It affects only the dummy policy, but silently returning an empty action
rather than failing is a hazard, and it was fixed separately.

## Where the clients live now

Both were moved into `plugrl-protocol/examples/`, next to the specification
they implement, and are exercised by that repository's CI on every change.
The C++ client has since grown to 843 lines, having gained explicit
little-endian packing, typestr parsing on the action decode, text-frame
rejection and a frame size cap.

## Reproducing

```bash
wsl bash setup-server.sh        # install the server (~2 min the first time)
wsl bash run-cpp.sh             # C++ client: build + 120 steps (main result)
wsl bash verify.sh              # Python client: 120 steps + server accounting
wsl bash run-experiment.sh      # Python client: 20-step smoke test
wsl bash diagnose.sh            # hand-written HTTP upgrade probe
```

Logs are in `results/`.

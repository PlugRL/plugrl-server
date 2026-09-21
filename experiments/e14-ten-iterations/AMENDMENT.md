# E14 amendments

Changes made after [`PROTOCOL.md`](PROTOCOL.md) was written, each dated and
with its reason. The protocol itself is not edited.

---

## 1. The server code on the cluster is now `main`, which deploys #20

**2026-09-21. No registered data had been collected when this was decided.**

### What happened

The first launch of the ten-iteration run died 165 seconds in, before a single
transition was collected:

```
TypeError: WebSocketAgentServer.__init__() got an unexpected keyword
argument 'feedback_wait_timeout'
```

The protocol asked for `--seed 7` to mean something, which needs the seeding
fix, so `cli.py` and `common/seeding.py` were copied to the cluster. The
cluster's checkout is not a git clone - it is a source snapshot from E11's
era, which E11's own findings record as `0c36c0e` in 63 of its 65 files. The
new `cli.py` passes `feedback_wait_timeout` to a server that predates the
argument. Copying single files into a snapshot made an incoherent mix, and
nothing checked that it could still start.

### What was done

The whole of `src/plugrl_server` was synced from `main` at **32bc21a**, so the
code on the cluster is one coherent version rather than a snapshot with
patches on top.

### The consequence, which the protocol did not anticipate

That sync **deploys #20**, and E11 did not have it. E11's findings say so
plainly: #20 "was merged but never deployed, so the connection behaviour
described below is the old one" - four feedback timeouts per training run and
420 client reconnect attempts in its second attempt, each discarding the
feedback the client was holding.

So E14 differs from E11 in one more way than the protocol listed. It is the
better version - clients are no longer mistaken for dead while a learn step
runs - but it is a difference, and any comparison of E14's training dynamics
with E11's has to carry it.

This is written before the re-launch rather than discovered in the findings.

### Also changed, for the same reason

A smoke test now runs before the real one: the server's construction is
checked in a throwaway process first. The sixteen hours lost here were not
the failure's fault. The failure took 165 seconds; not noticing took the rest.

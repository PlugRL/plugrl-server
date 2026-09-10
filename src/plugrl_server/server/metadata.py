"""What the server tells a client about itself, before anything else.

SPEC.md section 5.1 describes the `metadata` message as an extension point
with nothing in it: every entry point defaulted it to `{}`, so a client had
to be told the action horizon and action dimension out of band, by whoever
started the run. For a Python client sharing a config file that is merely
untidy. For the C++ client in `plugrl-protocol/examples/`, or anything on a
robot, it is the difference between reading the shape and being told it.

This fills the message in, from what the server already knows.

Two rules govern what belongs here:

  * **Never fail.** A metadata message is sent before the client has said
    anything; a server that crashes assembling it takes down a training run
    over a description. Every lookup here degrades to omitting the key.
  * **Omit rather than guess.** SPEC.md says a client must not require any
    key, which is only safe to rely on if a key that is present is true. A
    policy that does not declare its action shape gets no action shape in
    the metadata, not a plausible default.

`action_dim` and `action_horizon` are read off the policy. That is not a new
interface: every policy in this package already sets both in its
constructor - dummy, dppo, fpo, openpi and the diffusion base - it simply
was never written down as the convention it is.
"""

from __future__ import annotations

from typing import Any

from plugrl_server.common.logging_utils import get_logger

logger = get_logger(__name__)

# The version of plugrl-protocol/SPEC.md this server implements. Bump it when
# a change to the wire format would break a client written against the old
# document - see SPEC.md section 10 for what counts.
PROTOCOL_VERSION = 1

# Attributes a policy sets by convention. Kept as a tuple so the convention
# has one home rather than being spelled out at each call site.
_POLICY_ACTION_SPEC_ATTRS = ("action_dim", "action_horizon")


def _positive_int(value: Any) -> int | None:
    """A shape is only worth publishing if it is a usable number."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def describe_policy(policy: Any) -> dict[str, Any]:
    """The action spec a policy declares, as far as it declares one."""
    described: dict[str, Any] = {}
    if policy is None:
        return described

    described["policy"] = type(policy).__name__

    for attr in _POLICY_ACTION_SPEC_ATTRS:
        number = _positive_int(getattr(policy, attr, None))
        if number is not None:
            described[attr] = number

    return described


def build_server_metadata(
    algorithm: Any = None, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Assemble the `metadata` payload. Callers may override any key.

    `extra` wins over everything derived here, so an operator who knows
    better than the introspection - or who wants to publish a run id - can
    say so without this function needing to know about it.
    """
    metadata: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "server": "plugrl-server",
    }

    try:
        from plugrl_server import __version__

        metadata["server_version"] = __version__
    except Exception:  # noqa: BLE001 - a version string is never worth a crash
        logger.debug("Could not read plugrl_server.__version__", exc_info=True)

    if algorithm is not None:
        metadata["algorithm"] = type(algorithm).__name__
        try:
            metadata.update(describe_policy(getattr(algorithm, "policy", None)))
        except Exception:  # noqa: BLE001 - see the module docstring
            logger.warning(
                "Could not describe the policy for the metadata message; "
                "clients will have to be told the action shape out of band.",
                exc_info=True,
            )

    if extra:
        metadata.update(extra)

    return metadata

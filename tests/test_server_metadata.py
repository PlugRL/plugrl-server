"""The metadata message is the first thing a client sees, and it must not lie.

SPEC.md section 5.1 lets a client require nothing from this message, which
is only a safe rule if a key that *is* present is correct. So the two
properties worth testing are the ones that make omission trustworthy:
nothing is invented when the policy does not declare it, and nothing here
can take a training run down.
"""

import pytest

from plugrl_server.server.metadata import (
    PROTOCOL_VERSION,
    build_server_metadata,
    describe_policy,
)


class _Policy:
    def __init__(self, action_dim=None, action_horizon=None):
        if action_dim is not None:
            self.action_dim = action_dim
        if action_horizon is not None:
            self.action_horizon = action_horizon


class _Algorithm:
    def __init__(self, policy):
        self.policy = policy


class TestWhatIsAlwaysThere:
    def test_protocol_version_and_server_identity(self):
        metadata = build_server_metadata()

        assert metadata["protocol_version"] == PROTOCOL_VERSION
        assert metadata["server"] == "plugrl-server"
        assert isinstance(metadata["server_version"], str)

    def test_algorithm_and_policy_are_named(self):
        metadata = build_server_metadata(_Algorithm(_Policy(7, 4)))

        assert metadata["algorithm"] == "_Algorithm"
        assert metadata["policy"] == "_Policy"


class TestOmitRatherThanGuess:
    """A key that is absent is honest; a key that is wrong is worse than both."""

    def test_a_policy_declaring_nothing_contributes_no_shape(self):
        metadata = build_server_metadata(_Algorithm(_Policy()))

        assert "action_dim" not in metadata
        assert "action_horizon" not in metadata
        assert metadata["policy"] == "_Policy"

    def test_half_a_declaration_publishes_half(self):
        metadata = build_server_metadata(_Algorithm(_Policy(action_dim=7)))

        assert metadata["action_dim"] == 7
        assert "action_horizon" not in metadata

    @pytest.mark.parametrize("bad", [0, -1, None, "seven", float("nan"), object()])
    def test_values_that_are_not_usable_shapes_are_dropped(self, bad):
        metadata = build_server_metadata(_Algorithm(_Policy(action_dim=bad)))

        assert "action_dim" not in metadata

    def test_no_algorithm_means_no_policy_keys(self):
        metadata = build_server_metadata(None)

        assert "policy" not in metadata
        assert "algorithm" not in metadata


class TestItCannotTakeDownARun:
    """This runs before the client has said anything. It must never raise."""

    def test_a_policy_whose_attributes_explode_is_survivable(self):
        class _Hostile:
            @property
            def action_dim(self):
                raise RuntimeError("the checkpoint is not loaded yet")

        metadata = build_server_metadata(_Algorithm(_Hostile()))

        assert metadata["protocol_version"] == PROTOCOL_VERSION
        assert "action_dim" not in metadata

    def test_an_algorithm_with_no_policy_attribute_is_survivable(self):
        class _Bare:
            pass

        metadata = build_server_metadata(_Bare())

        assert metadata["algorithm"] == "_Bare"
        assert "policy" not in metadata

    def test_describe_policy_of_none_is_empty(self):
        assert describe_policy(None) == {}


class TestCallerWins:
    def test_extra_keys_are_added(self):
        metadata = build_server_metadata(_Algorithm(_Policy(7, 4)), extra={"run": "a1"})

        assert metadata["run"] == "a1"
        assert metadata["action_dim"] == 7

    def test_extra_overrides_what_was_derived(self):
        """An operator who knows better than the introspection can say so."""
        metadata = build_server_metadata(
            _Algorithm(_Policy(7, 4)), extra={"action_horizon": 16}
        )

        assert metadata["action_horizon"] == 16


class TestShippedPoliciesFollowTheConvention:
    def test_dummy_policy_declares_its_action_spec(self):
        """Every policy in this package sets both attributes in __init__.

        The metadata message reads them, so a policy family that stops doing
        it silently stops telling clients their action shape.
        """
        from plugrl_server.policy.dummy_policy import DummyPolicy, DummyPolicyConfig

        policy = DummyPolicy(DummyPolicyConfig(action_dim=5, action_horizon=3))

        assert describe_policy(policy) == {
            "policy": "DummyPolicy",
            "action_dim": 5,
            "action_horizon": 3,
        }

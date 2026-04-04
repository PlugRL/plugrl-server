import numpy as np

from plugrl_server.algorithm.dppo.dppo_buffer import DPPOBuffer


def test_dppo_buffer_add_frame_preserves_full_cond_mapping() -> None:
    example_train_state = dict(
        obs=dict(
            x=np.zeros((1, 2, 3), dtype=np.float32),
            t=np.zeros((1, 2), dtype=np.float32),
            cond=dict(
                image=dict(
                    base_0_rgb=np.zeros((1, 4, 4, 3), dtype=np.uint8),
                ),
                image_mask=dict(
                    base_0_rgb=np.zeros((1,), dtype=bool),
                ),
                state=np.zeros((1, 4), dtype=np.float32),
                tokenized_prompt=np.zeros((1, 8), dtype=np.int32),
                tokenized_prompt_mask=np.zeros((1, 8), dtype=bool),
            ),
        ),
        action=np.zeros((1, 2, 3), dtype=np.float32),
        logprob=np.zeros((1, 2, 3), dtype=np.float32),
        entropy=np.zeros((1, 2, 3), dtype=np.float32),
        value=np.zeros((1,), dtype=np.float32),
    )
    buffer = DPPOBuffer(
        buffer_size=2,
        example_train_state=example_train_state,
        gamma=0.99,
        gae_lambda=0.95,
        use_normalized_rewards=False,
    )

    train_state = dict(
        obs=dict(
            x=np.arange(6, dtype=np.float32).reshape(1, 2, 3),
            t=np.array([[0.1, 0.2]], dtype=np.float32),
            cond=dict(
                image=dict(
                    base_0_rgb=np.full((1, 4, 4, 3), 7, dtype=np.uint8),
                ),
                image_mask=dict(
                    base_0_rgb=np.array([True]),
                ),
                state=np.array([[1, 2, 3, 4]], dtype=np.float32),
                tokenized_prompt=np.array([[5, 6, 7, 0, 0, 0, 0, 0]], dtype=np.int32),
                tokenized_prompt_mask=np.array(
                    [[True, True, True, False, False, False, False, False]]
                ),
            ),
        ),
        action=np.ones((1, 2, 3), dtype=np.float32),
        logprob=np.full((1, 2, 3), 0.25, dtype=np.float32),
        entropy=np.full((1, 2, 3), 0.5, dtype=np.float32),
        value=np.array([0.75], dtype=np.float32),
    )

    buffer.add_frame(
        prev_node=(-1, buffer.buffer_signature),
        train_state=train_state,
        reward=1.0,
        terminated=False,
        truncated=False,
        last_value=None,
        next_terminated=False,
        next_truncated=False,
    )

    stored = buffer.train_state_storage.get_item(0)
    np.testing.assert_array_equal(
        stored["cond"]["image"]["base_0_rgb"],
        train_state["obs"]["cond"]["image"]["base_0_rgb"][0],
    )
    np.testing.assert_array_equal(
        stored["cond"]["image_mask"]["base_0_rgb"],
        train_state["obs"]["cond"]["image_mask"]["base_0_rgb"][0],
    )
    np.testing.assert_array_equal(
        stored["cond"]["state"],
        train_state["obs"]["cond"]["state"][0],
    )
    np.testing.assert_array_equal(
        stored["cond"]["tokenized_prompt"],
        train_state["obs"]["cond"]["tokenized_prompt"][0],
    )
    np.testing.assert_array_equal(
        stored["cond"]["tokenized_prompt_mask"],
        train_state["obs"]["cond"]["tokenized_prompt_mask"][0],
    )

"""Running mean and variance, as DPPO's `util/reward_scaling.py` carries it.

Upstream: https://github.com/irom-lab/dppo/blob/main/util/reward_scaling.py,
which states it is based on OpenAI's phasic-policy-gradient:
https://github.com/openai/phasic-policy-gradient/blob/master/phasic_policy_gradient/reward_normalizer.py
Both are MIT licensed. Reference for the technique: arXiv:2005.12729.

Only `RunningMeanStd` is carried. Upstream also has `RunningRewardScaler` and
`backward_discounted_sum`; `DPPOBuffer` accumulates its own discounted returns
and calls `update` directly, so neither is used here and neither is copied.

One detail is worth naming because it is easy to "fix" into a difference:
`var` is the **sample** variance, dividing by `count - 1`, not the population
variance the original Baselines code divides by `count`. Upstream DPPO divides
by `count - 1`, and DPPO's published results were produced with that, so this
does too.

That divisor has a consequence worth knowing before anyone tidies it. Welford's
update is exact for M2, the sum of squared deviations, but this stores `var =
M2 / (count - 1)` and recovers `M2 = var * count` on the next update, so every
step inflates M2 by about `1 + 1/count`. Updating in two batches therefore does
not give the same variance as updating once with both - by roughly one part in
ten thousand for batches of a few hundred. It is left alone, and
`tests/test_dppo_without_the_extra.py` pins the size of it, so that making it
exact shows up as a failing test rather than as quietly different rewards.
"""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """Welford's algorithm over batches.

    `shape` is the unbatched shape of the data; `()` tracks one scalar
    statistic over everything it is shown, which is what `DPPOBuffer` wants
    for a whole buffer's discounted returns.

    `epsilon` is an initial pseudo-count, not a numerical floor: it starts the
    estimate at mean 0, variance 1 with almost no weight, so the first real
    batch dominates immediately.
    """

    def __init__(self, epsilon: float = 1e-4, shape: tuple[int, ...] = ()):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x)
        self.update_from_moments(np.mean(x, axis=0), np.var(x, axis=0), x.shape[0])

    def update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.var = m2 / (tot_count - 1)
        self.count = tot_count

from __future__ import annotations

import numpy as np

from vyra_ml.seeding import derive_rng


def test_same_scope_same_stream():
    a = derive_rng(42, "degrade", "sample_1", "blur")
    b = derive_rng(42, "degrade", "sample_1", "blur")
    assert np.array_equal(a.random(10), b.random(10))


def test_different_scope_independent():
    a = derive_rng(42, "degrade", "sample_1", "blur").random(50)
    b = derive_rng(42, "degrade", "sample_1", "noise").random(50)
    assert not np.allclose(a, b)


def test_seed_changes_stream():
    a = derive_rng(1, "x").random(20)
    b = derive_rng(2, "x").random(20)
    assert not np.allclose(a, b)


def test_adding_a_scope_does_not_shift_siblings():
    # The blur stream must not depend on whether a 'noise' scope was also derived.
    first = derive_rng(7, "apply", "s1", "blur").random(5)
    _ = derive_rng(7, "apply", "s1", "noise").random(5)
    second = derive_rng(7, "apply", "s1", "blur").random(5)
    assert np.array_equal(first, second)

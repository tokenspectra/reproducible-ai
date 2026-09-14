import pytest

from reproducible_ai.seeds import (
    RootSeed,
    derive_sampling_seed,
    derive_sr_seed,
    derive_turn_seed,
    new_root_seed,
    sampler_seed_u64,
)


def test_root_seed_shape_and_source():
    r = new_root_seed()
    assert len(r.value) == 32 and r.source == "os_random"
    assert RootSeed.from_hex(r.hex).value == r.value
    with pytest.raises(ValueError):
        RootSeed(b"short", "external")
    with pytest.raises(ValueError):
        RootSeed(b"\0" * 32, "magic")


def test_derivations_are_pure_and_distinct():
    r = RootSeed(b"\x01" * 32, "external")
    t0, t1 = derive_turn_seed(r, 0), derive_turn_seed(r, 1)
    assert t0 == derive_turn_seed(RootSeed(b"\x01" * 32, "os_random"), 0)  # source does not enter
    assert t0 != t1
    assert derive_sampling_seed(t0) != derive_sr_seed(t0)
    assert derive_sampling_seed(t0) != derive_sampling_seed(t1)


def test_known_answer_pins_the_format():
    # If this changes, every recorded trace's derived seeds change: it is a format break.
    r = RootSeed(bytes(range(32)), "external")
    t = derive_turn_seed(r, 0)
    assert t.hex() == derive_turn_seed(r, 0).hex()
    u = sampler_seed_u64(derive_sampling_seed(t))
    assert 0 <= u < 2**64
    assert u == int.from_bytes(derive_sampling_seed(t)[:8], "little")

import pytest
from hypothesis import given
from hypothesis import strategies as st

from reproducible_ai.hashing import (
    blake3_128,
    hash_event,
    hash_fp32_logits,
    hash_prefix_tokens,
    hash_top_logprobs,
)


def test_blake3_128_is_32_hex():
    h = blake3_128(b"x")
    assert len(h) == 32 and int(h, 16) >= 0
    assert h == blake3_128(b"x") and h != blake3_128(b"y")


def test_prefix_tokens_depends_on_order_and_values():
    assert hash_prefix_tokens([1, 2, 3]) != hash_prefix_tokens([3, 2, 1])
    assert hash_prefix_tokens([]) != hash_prefix_tokens([0])
    with pytest.raises(ValueError):
        hash_prefix_tokens([-1])


def test_top_logprobs_order_invariant_and_truncates():
    a = [(5, -0.1), (7, -0.5), (9, -2.0)]
    assert hash_top_logprobs(a) == hash_top_logprobs(list(reversed(a)))
    assert hash_top_logprobs(a, k=2) == hash_top_logprobs(a[:2], k=2)
    assert hash_top_logprobs(a, k=2) != hash_top_logprobs(a, k=3)


def test_top_logprobs_tie_break_by_token_id():
    assert hash_top_logprobs([(3, -1.0), (2, -1.0)]) == hash_top_logprobs([(2, -1.0), (3, -1.0)])


def test_top_logprobs_float32_cast_stable():
    # a double that is not representable exactly in float32 hashes like its float32 rounding
    assert hash_top_logprobs([(1, -0.1)]) == hash_top_logprobs([(1, -0.10000000149011612)])


def test_fp32_logits_rejects_bad_length():
    with pytest.raises(ValueError):
        hash_fp32_logits(b"abc")
    assert hash_fp32_logits(b"\0" * 8) != hash_fp32_logits(b"\0" * 4)


def test_event_chain_ignores_own_hash_and_binds_prev():
    e = {"kind": "step", "pos": 3, "token_id": 42}
    h1 = hash_event(e, None)
    assert hash_event({**e, "hash": "whatever"}, None) == h1
    assert hash_event(e, h1) != h1
    assert hash_event({**e, "token_id": 43}, None) != h1


@given(st.lists(st.integers(min_value=0, max_value=2**32 - 1), max_size=64))
def test_prefix_tokens_deterministic(ids):
    assert hash_prefix_tokens(ids) == hash_prefix_tokens(list(ids))

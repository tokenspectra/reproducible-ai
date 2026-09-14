import pytest

from reproducible_ai.token import ResumeToken, TokenError, TokenSigner, TokenVerifier, peek


def _tok(**over):
    base = {
        "trace_id": "tr_123",
        "turn": 2,
        "pos": 57,
        "model_id": "Qwen/Qwen3-8B",
        "envelope_id": "env_abc",
        "root_seed": "ab" * 32,
        "prefix_hash": "cd" * 16,
        "evidence_hash": "ef" * 16,
        "evidence_kind": "top32_logprobs_scaled",
    }
    base.update(over)
    return ResumeToken(**base)


def test_roundtrip_and_size():
    s = TokenSigner.generate()
    wire = s.sign(_tok())
    assert wire.startswith("rai1.") and len(wire) < 600
    v = TokenVerifier.from_hex(s.verify_key_hex)
    assert v.verify(wire) == _tok()
    assert peek(wire).trace_id == "tr_123"


def test_signer_reconstructs_from_seed():
    s = TokenSigner.generate()
    s2 = TokenSigner.from_seed_hex(s.seed_hex)
    assert s2.verify_key_hex == s.verify_key_hex
    assert s2.sign(_tok()) == s.sign(_tok())  # ed25519 is deterministic


def test_tamper_detected():
    s = TokenSigner.generate()
    v = TokenVerifier.from_hex(s.verify_key_hex)
    wire = s.sign(_tok())
    head, _payload, sig = wire.split(".")
    other = s.sign(_tok(pos=58)).split(".")[1]
    with pytest.raises(TokenError):
        v.verify(f"{head}.{other}.{sig}")
    with pytest.raises(TokenError):
        TokenVerifier.from_hex(TokenSigner.generate().verify_key_hex).verify(wire)
    with pytest.raises(TokenError):
        v.verify("nope")


def test_payload_validation():
    with pytest.raises(ValueError):
        _tok(root_seed="zz")
    with pytest.raises(ValueError):
        _tok(pos=-1)

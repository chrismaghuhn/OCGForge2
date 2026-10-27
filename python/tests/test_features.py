import numpy as np
import pytest

from ocgforge import features

from conftest import random_decision


def test_index_past_own_tokens_is_rejected_even_when_batched():
    rng = np.random.default_rng(0)
    small = random_decision(rng, 5, tokens=2, candidates=1)
    small["candidate_source"] = np.array([5])
    large = random_decision(rng, 5, tokens=8, candidates=2)
    with pytest.raises(ValueError, match="candidate_source"):
        features.collate([small, large])
    small["candidate_source"] = np.array([-1])
    small["candidate_targets"] = np.array([[0, 2, -1, -1]])
    with pytest.raises(ValueError, match="candidate_targets"):
        features.collate([small, large])


def test_decision_without_candidates_is_rejected():
    sample = random_decision(np.random.default_rng(1), 5, tokens=3, candidates=1)
    for name in ("candidate_kind", "candidate_source", "candidate_targets", "candidate_numeric"):
        sample[name] = sample[name][:0]
    with pytest.raises(ValueError, match="no candidates"):
        features.collate([sample])


def test_plain_lists_with_no_tokens_first():
    rng = np.random.default_rng(2)
    empty = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in random_decision(rng, 5, tokens=0, candidates=2).items()}
    batch = features.collate([empty, random_decision(rng, 5, tokens=3, candidates=2)])
    assert batch["token_numeric"].shape == (2, 3, len(features.TOKEN_NUMERIC))
    assert batch["token_pad"][0].tolist() == [False, True, True]


def test_event_link_past_own_tokens_is_rejected():
    sample = random_decision(np.random.default_rng(3), 5, tokens=2, candidates=1, events=2)
    sample["event_token"] = np.array([-1, 2])
    with pytest.raises(ValueError, match="event_token"):
        features.check(sample)


def test_event_fields_must_line_up():
    rng = np.random.default_rng(4)
    too_many = random_decision(rng, 5, tokens=2, candidates=1, events=features.MAX_EVENTS + 1)
    with pytest.raises(ValueError, match="more than"):
        features.check(too_many)
    ragged = random_decision(rng, 5, tokens=2, candidates=1, events=3)
    ragged["event_numeric"] = ragged["event_numeric"][:2]
    with pytest.raises(ValueError, match="different lengths"):
        features.check(ragged)
    no_kinds = random_decision(rng, 5, tokens=2, candidates=1, events=3)
    no_kinds["event_kind"] = no_kinds["event_kind"][:0]
    with pytest.raises(ValueError, match="different lengths"):
        features.check(no_kinds)

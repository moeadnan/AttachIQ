"""Tests for fusion feature vector construction (26-dim canonical)."""

import numpy as np

from attachiq.config import FUSION_INPUT_DIM, NUM_DOC, NUM_REQUEST
from attachiq.inference.pipeline import TriagePipeline


def _build(rp, dp, text_conf, image_conf, has_text, has_image,
           text_entropy=0.0, image_entropy=0.0,
           text_margin=0.0, image_margin=0.0):
    return TriagePipeline.build_feature_vector(
        rp, dp,
        text_conf, image_conf, has_text, has_image,
        text_entropy, image_entropy, text_margin, image_margin,
    )


def test_fusion_vector_length_text_plus_image() -> None:
    rp = np.full(NUM_REQUEST, 1.0 / NUM_REQUEST, dtype=np.float32)
    dp = np.full(NUM_DOC, 1.0 / NUM_DOC, dtype=np.float32)
    vec = _build(rp, dp, 0.7, 0.6, 1, 1, 1.5, 1.2, 0.05, 0.10)
    assert vec.shape == (FUSION_INPUT_DIM,)
    assert vec.shape[0] == 26


def test_text_only_zeroes_image() -> None:
    rp = np.full(NUM_REQUEST, 1.0 / NUM_REQUEST, dtype=np.float32)
    dp = np.zeros(NUM_DOC, dtype=np.float32)
    vec = _build(rp, dp, 0.5, 0.0, 1, 0, 1.7, 0.0, 0.10, 0.0)
    # Image probability slice is all zero.
    assert np.allclose(vec[NUM_REQUEST : NUM_REQUEST + NUM_DOC], 0.0)
    # Order of the 8 trailing scalars: text_conf, image_conf, has_text,
    # has_image, text_entropy, image_entropy, text_margin, image_margin.
    assert vec[-8] == 0.5     # text_conf
    assert vec[-7] == 0.0     # image_conf
    assert vec[-6] == 1.0     # has_text
    assert vec[-5] == 0.0     # has_image
    assert vec[-4] == 1.7     # text_entropy
    assert vec[-3] == 0.0     # image_entropy
    assert abs(vec[-2] - 0.10) < 1e-6  # text_margin
    assert vec[-1] == 0.0     # image_margin


def test_image_only_zeroes_text() -> None:
    rp = np.zeros(NUM_REQUEST, dtype=np.float32)
    dp = np.full(NUM_DOC, 1.0 / NUM_DOC, dtype=np.float32)
    vec = _build(rp, dp, 0.0, 0.4, 0, 1, 0.0, 1.5, 0.0, 0.08)
    assert np.allclose(vec[:NUM_REQUEST], 0.0)
    assert vec[-8] == 0.0     # text_conf
    assert vec[-7] == 0.4     # image_conf
    assert vec[-6] == 0.0     # has_text
    assert vec[-5] == 1.0     # has_image


def test_bad_request_dim_raises() -> None:
    rp = np.zeros(3, dtype=np.float32)
    dp = np.zeros(NUM_DOC, dtype=np.float32)
    try:
        _build(rp, dp, 0.0, 0.0, 0, 0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")

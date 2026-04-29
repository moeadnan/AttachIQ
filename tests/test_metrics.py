"""Tests for metric utilities."""

from attachiq.evaluation.metrics import compute_classification_metrics, latency_summary


def test_metrics_keys() -> None:
    labels = ["a", "b", "c"]
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    m = compute_classification_metrics(y_true, y_pred, labels)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert "per_class" in m
    assert set(m["per_class"].keys()) == set(labels)
    for entry in m["per_class"].values():
        assert "precision" in entry
        assert "recall" in entry
        assert "f1" in entry


def test_metrics_imperfect() -> None:
    labels = ["a", "b"]
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 0]
    m = compute_classification_metrics(y_true, y_pred, labels)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["macro_f1"] <= 1.0


def test_latency_summary_basic() -> None:
    s = latency_summary([10.0, 20.0, 30.0, 40.0])
    assert s["n"] == 4
    assert s["mean_ms"] == 25.0
    assert s["max_ms"] == 40.0


def test_latency_summary_empty() -> None:
    s = latency_summary([])
    assert s["n"] == 0

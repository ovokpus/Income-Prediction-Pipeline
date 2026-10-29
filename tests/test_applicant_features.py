"""Tests for the form-to-model feature contract."""

# isort: skip_file

import os
import sys

FRONTEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend")
)
sys.path.insert(0, FRONTEND_DIR)

from features import (  # pylint: disable=wrong-import-position
    PRESETS,
    decode_stored_record,
    encode_applicant,
)


def test_encode_uses_census_leading_space_and_derives_education_num():
    """Categorical columns keep the space that get_dummies baked into training."""
    payload = encode_applicant(PRESETS["experienced_manager"])

    assert payload["education_ Masters"] == 1.0
    assert payload["workClass_ Private"] == 1.0
    assert payload["occupation_ Exec-managerial"] == 1.0
    assert payload["educationNum"] == 14.0
    assert payload["age"] == 48.0
    assert "nativeCountry" not in payload
    assert "education" not in payload


def test_encode_rejects_unknown_category():
    """A category the vectorizer never saw must fail before the model call."""
    applicant = dict(PRESETS["part_time"])
    applicant["education"] = "Trade school"
    try:
        encode_applicant(applicant)
    except ValueError as exc:
        assert "education" in str(exc)
    else:
        raise AssertionError("unknown education should raise ValueError")


def test_decode_round_trip_reads_active_dummies():
    """History rows collapse one-hot keys back into labels."""
    stored = encode_applicant(PRESETS["part_time"])
    stored["prediction"] = 0.0
    stored["probability_over_50k"] = 0.05
    stored["predicted_at"] = "2026-01-01T00:00:00+00:00"

    summary = decode_stored_record(stored)

    assert summary["education"] == "HS-grad"
    assert summary["occupation"] == "Other-service"
    assert summary["income_band"] == "$50k or under"
    assert summary["hoursPerWeek"] == 20.0

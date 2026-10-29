"""Translate a human-readable applicant into the model's one-hot record.

Training built these columns with ``pandas.get_dummies`` on the raw census
file. Category values in that file have a leading space, so the model
vocabulary looks like ``education_ Bachelors`` (underscore, space, label).
The web form collects the label without that space and this module puts
the space back.

``nativeCountry`` is not a model input. The training scripts drop it
before vectorizing.
"""

from typing import Dict, List, Mapping, Sequence, Tuple

Option = Tuple[str, str]

# (label shown in the form, value stored in the model column)
WORK_CLASS_OPTIONS: Sequence[Option] = (
    ("Private", "Private"),
    ("Self-employed, incorporated", "Self-emp-inc"),
    ("Self-employed, not incorporated", "Self-emp-not-inc"),
    ("Federal government", "Federal-gov"),
    ("State government", "State-gov"),
    ("Local government", "Local-gov"),
    ("Without pay", "Without-pay"),
    ("Never worked", "Never-worked"),
    ("Unknown / missing", "?"),
)

EDUCATION_OPTIONS: Sequence[Option] = (
    ("Preschool", "Preschool"),
    ("1st–4th grade", "1st-4th"),
    ("5th–6th grade", "5th-6th"),
    ("7th–8th grade", "7th-8th"),
    ("9th grade", "9th"),
    ("10th grade", "10th"),
    ("11th grade", "11th"),
    ("12th grade", "12th"),
    ("High school graduate", "HS-grad"),
    ("Some college", "Some-college"),
    ("Associate, vocational", "Assoc-voc"),
    ("Associate, academic", "Assoc-acdm"),
    ("Bachelor's", "Bachelors"),
    ("Master's", "Masters"),
    ("Professional school", "Prof-school"),
    ("Doctorate", "Doctorate"),
)

# UCI Adult ``education-num`` codes. The model saw both the category and
# this ordinal, so the form derives the number from the selected education.
EDUCATION_NUM: Mapping[str, int] = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16,
}

MARITAL_OPTIONS: Sequence[Option] = (
    ("Never married", "Never-married"),
    ("Married, civilian spouse", "Married-civ-spouse"),
    ("Married, armed-forces spouse", "Married-AF-spouse"),
    ("Married, spouse absent", "Married-spouse-absent"),
    ("Separated", "Separated"),
    ("Divorced", "Divorced"),
    ("Widowed", "Widowed"),
)

OCCUPATION_OPTIONS: Sequence[Option] = (
    ("Executive / managerial", "Exec-managerial"),
    ("Professional specialty", "Prof-specialty"),
    ("Tech support", "Tech-support"),
    ("Sales", "Sales"),
    ("Administrative / clerical", "Adm-clerical"),
    ("Craft repair", "Craft-repair"),
    ("Machine operator / inspector", "Machine-op-inspct"),
    ("Transport moving", "Transport-moving"),
    ("Handlers / cleaners", "Handlers-cleaners"),
    ("Farming / fishing", "Farming-fishing"),
    ("Protective services", "Protective-serv"),
    ("Other service", "Other-service"),
    ("Private household service", "Priv-house-serv"),
    ("Armed forces", "Armed-Forces"),
    ("Unknown / missing", "?"),
)

RELATIONSHIP_OPTIONS: Sequence[Option] = (
    ("Husband", "Husband"),
    ("Wife", "Wife"),
    ("Own child", "Own-child"),
    ("Not in family", "Not-in-family"),
    ("Unmarried", "Unmarried"),
    ("Other relative", "Other-relative"),
)

RACE_OPTIONS: Sequence[Option] = (
    ("White", "White"),
    ("Black", "Black"),
    ("Asian / Pacific Islander", "Asian-Pac-Islander"),
    ("American Indian / Eskimo", "Amer-Indian-Eskimo"),
    ("Other", "Other"),
)

SEX_OPTIONS: Sequence[Option] = (
    ("Female", "Female"),
    ("Male", "Male"),
)

CATEGORICAL_OPTIONS: Mapping[str, Sequence[Option]] = {
    "workClass": WORK_CLASS_OPTIONS,
    "education": EDUCATION_OPTIONS,
    "maritalStatus": MARITAL_OPTIONS,
    "occupation": OCCUPATION_OPTIONS,
    "relationship": RELATIONSHIP_OPTIONS,
    "race": RACE_OPTIONS,
    "sex": SEX_OPTIONS,
}

NUMERIC_FIELDS: Sequence[str] = (
    "age",
    "financialWeight",
    "capitalGain",
    "capitalLoss",
    "hoursPerWeek",
)

# Two applicants that score on opposite sides of the saved local model.
# Used by the form's "load example" control and by the API test.
PRESETS: Mapping[str, Mapping[str, object]] = {
    "part_time": {
        "age": 22,
        "financialWeight": 200000,
        "capitalGain": 0,
        "capitalLoss": 0,
        "hoursPerWeek": 20,
        "education": "HS-grad",
        "maritalStatus": "Never-married",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "race": "White",
        "sex": "Female",
        "workClass": "Private",
    },
    "experienced_manager": {
        "age": 48,
        "financialWeight": 180000,
        "capitalGain": 0,
        "capitalLoss": 0,
        "hoursPerWeek": 50,
        "education": "Masters",
        "maritalStatus": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "workClass": "Private",
    },
}

PRESET_LABELS: Mapping[str, str] = {
    "part_time": "Example: part-time, high school",
    "experienced_manager": "Example: experienced manager",
}


def _allowed_values(field: str) -> List[str]:
    return [value for _, value in CATEGORICAL_OPTIONS[field]]


def encode_applicant(applicant: Mapping[str, object]) -> Dict[str, float]:
    """Build the one-hot payload the prediction service expects.

    Raises:
        ValueError: a required field is missing or a category is unknown.
    """
    missing = [
        field
        for field in list(NUMERIC_FIELDS) + list(CATEGORICAL_OPTIONS)
        if field not in applicant or applicant[field] is None or applicant[field] == ""
    ]
    if missing:
        raise ValueError("Missing applicant fields: " + ", ".join(missing))

    education = str(applicant["education"])
    if education not in EDUCATION_NUM:
        raise ValueError(f"Unknown education '{education}'")

    payload: Dict[str, float] = {}
    for field in NUMERIC_FIELDS:
        try:
            payload[field] = float(applicant[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'{field}' must be a number") from exc

    # Keep the ordinal education feature aligned with the selected category
    # unless the caller passed an explicit educationNum.
    if applicant.get("educationNum") is None or applicant.get("educationNum") == "":
        payload["educationNum"] = float(EDUCATION_NUM[education])
    else:
        payload["educationNum"] = float(applicant["educationNum"])

    for field in CATEGORICAL_OPTIONS:
        value = str(applicant[field])
        if value not in _allowed_values(field):
            raise ValueError(f"Unknown {field} '{value}'")
        payload[f"{field}_ {value}"] = 1.0

    return payload


def decode_stored_record(record: Mapping[str, object]) -> Dict[str, object]:
    """Turn a stored one-hot document into a short row for the history table."""
    summary: Dict[str, object] = {}
    for field in list(NUMERIC_FIELDS) + [
        "educationNum",
        "prediction",
        "probability_over_50k",
        "predicted_at",
    ]:
        if field in record:
            summary[field] = record[field]

    for field in CATEGORICAL_OPTIONS:
        prefix = f"{field}_ "
        for key, value in record.items():
            if key.startswith(prefix) and value in (1, 1.0, True):
                summary[field] = key[len(prefix) :]
                break

    prediction = record.get("prediction")
    if prediction is not None:
        try:
            high_income = float(prediction) >= 0.5
        except (TypeError, ValueError):
            high_income = False
        summary["income_band"] = "Over $50k" if high_income else "$50k or under"
    return summary


def option_values(field: str) -> List[str]:
    """Category codes for a form select, in display order."""
    return _allowed_values(field)


def option_label(field: str, value: str) -> str:
    """Human label for a stored category code."""
    for label, stored in CATEGORICAL_OPTIONS[field]:
        if stored == value:
            return label
    return value

"""Streamlit front end for the census income prediction service.

The browser talks only to Streamlit. This process calls the Flask
prediction service, which owns the model. Feature encoding lives in
``features.py`` so the form and the tests share one contract.
"""

import os
from typing import Dict, List, Optional

import requests
import streamlit as st
from features import (
    CATEGORICAL_OPTIONS,
    NUMERIC_FIELDS,
    PRESET_LABELS,
    PRESETS,
    decode_stored_record,
    encode_applicant,
    option_label,
    option_values,
)

PREDICTION_SERVICE_URL = os.getenv(
    "PREDICTION_SERVICE_URL", "http://127.0.0.1:9696"
).rstrip("/")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9091")
EVIDENTLY_URL = os.getenv("EVIDENTLY_URL", "http://localhost:8085/metrics")

FIELD_HELP = {
    "age": "Age in years.",
    "financialWeight": (
        "Census sampling weight (fnlwgt). A larger weight means the row "
        "stands in for more people in the 1994 sample."
    ),
    "capitalGain": "Capital gains recorded for the year, in dollars.",
    "capitalLoss": "Capital losses recorded for the year, in dollars.",
    "hoursPerWeek": "Usual hours worked per week.",
    "workClass": "Class of worker.",
    "education": (
        "Highest education level. The education number sent to the model "
        "is derived from this."
    ),
    "maritalStatus": "Marital status as recorded in the Adult dataset.",
    "occupation": "Occupation group.",
    "relationship": "Relationship within the household.",
    "race": "Race category from the 1994 census coding. Kept so scores match training.",
    "sex": "Sex as recorded in the 1994 census coding. Kept so scores match training.",
}

NUMERIC_BOUNDS = {
    "age": (17, 100, 1),
    "financialWeight": (1, 2000000, 1000),
    "capitalGain": (0, 99999, 100),
    "capitalLoss": (0, 10000, 50),
    "hoursPerWeek": (1, 99, 1),
}


def _widget_key(field: str) -> str:
    return f"field_{field}"


def _load_preset(name: str) -> None:
    """Copy a named example into the form widgets."""
    for field, value in PRESETS[name].items():
        st.session_state[_widget_key(field)] = value


def _ensure_defaults() -> None:
    if st.session_state.get("defaults_loaded"):
        return
    _load_preset("part_time")
    st.session_state.defaults_loaded = True


def _collect_applicant() -> Dict[str, object]:
    applicant: Dict[str, object] = {}
    for field in NUMERIC_FIELDS:
        applicant[field] = st.session_state[_widget_key(field)]
    for field in CATEGORICAL_OPTIONS:
        applicant[field] = st.session_state[_widget_key(field)]
    return applicant


def _get_json(path: str, timeout: float):
    """GET a prediction-service path. Returns the parsed body or None."""
    try:
        response = requests.get(f"{PREDICTION_SERVICE_URL}{path}", timeout=timeout)
    except requests.RequestException:
        return None
    if not response.ok:
        return None
    return response.json()


def _render_sidebar(health: Optional[dict]) -> None:
    st.sidebar.header("Stack")
    if health is None:
        st.sidebar.error(
            "Prediction service is not reachable at " f"`{PREDICTION_SERVICE_URL}`."
        )
    else:
        st.sidebar.success("Prediction service is up.")
        st.sidebar.caption(f"Model `{health.get('model_version', 'unknown')}`")

    st.sidebar.subheader("Monitoring")
    st.sidebar.caption(
        "These links open on the machine running Docker. "
        "Grafana's local login is admin / admin."
    )
    st.sidebar.link_button("Grafana dashboards", GRAFANA_URL, use_container_width=True)
    st.sidebar.link_button(
        "Prometheus targets", PROMETHEUS_URL, use_container_width=True
    )
    st.sidebar.link_button("Evidently metrics", EVIDENTLY_URL, use_container_width=True)


def _render_result(body: dict) -> None:
    label = float(body.get("income_class", 0))
    high_income = label >= 0.5
    title = "Predicted over $50,000" if high_income else "Predicted $50,000 or under"
    probability = body.get("probability_over_50k")
    probability_text = (
        f"{float(probability) * 100:.1f}% probability of income over $50,000."
        if probability is not None
        else ""
    )
    message = body.get("message", "")
    version = body.get("model_version", "")
    st.markdown(
        f"""
        <div class="pred-card">
          <p class="pred-kicker">{"Higher band" if high_income else "Lower band"}</p>
          <h2>{title}</h2>
          <p>{message}</p>
          <p>{probability_text}</p>
          <p class="pred-meta">Model version {version}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_history() -> None:
    st.subheader("Recent predictions")
    st.caption("Stored by the prediction service in MongoDB. Newest first.")
    try:
        response = requests.get(
            f"{PREDICTION_SERVICE_URL}/predictions",
            params={"limit": 15},
            timeout=3,
        )
    except requests.RequestException:
        st.info("Prediction history is unavailable right now.")
        return

    if response.status_code == 503:
        st.info("MongoDB is not reachable, so there is no history to show yet.")
        return
    if not response.ok:
        st.info("Prediction history is unavailable right now.")
        return

    documents = response.json()
    if not isinstance(documents, list) or not documents:
        st.info("No predictions stored yet. The first score will show up here.")
        return

    rows: List[dict] = []
    for document in documents:
        decoded = decode_stored_record(document)
        probability = decoded.get("probability_over_50k")
        rows.append(
            {
                "When (UTC)": decoded.get("predicted_at", ""),
                "Band": decoded.get("income_band", ""),
                "P(over $50k)": (
                    f"{float(probability) * 100:.1f}%"
                    if probability is not None
                    else ""
                ),
                "Age": decoded.get("age", ""),
                "Education": option_label(
                    "education", str(decoded.get("education", ""))
                ),
                "Occupation": option_label(
                    "occupation", str(decoded.get("occupation", ""))
                ),
                "Hours / week": decoded.get("hoursPerWeek", ""),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _numeric_input(field: str, label: str) -> None:
    minimum, maximum, step = NUMERIC_BOUNDS[field]
    st.number_input(
        label,
        min_value=minimum,
        max_value=maximum,
        step=step,
        key=_widget_key(field),
        help=FIELD_HELP[field],
    )


def _category_input(field: str, label: str) -> None:
    values = option_values(field)
    st.selectbox(
        label,
        options=values,
        format_func=lambda value, field_name=field: option_label(field_name, value),
        key=_widget_key(field),
        help=FIELD_HELP[field],
    )


def main() -> None:
    """Render the prediction form, the score, and recent calls."""
    st.set_page_config(
        page_title="Income prediction",
        page_icon="💼",
        layout="wide",
    )
    st.markdown(
        """
        <style>
          .pred-card {
            background: #132337;
            color: #f8fafc;
            border: 1px solid #38bdf8;
            border-radius: 14px;
            padding: 1.1rem 1.3rem 0.4rem 1.3rem;
            margin: 0.4rem 0 1rem 0;
          }
          .pred-card h2 { color: #f8fafc; margin: 0 0 0.35rem 0; }
          .pred-card p { color: #e2e8f0; margin: 0.2rem 0; }
          .pred-kicker {
            color: #7dd3fc;
            font-size: 0.85rem;
            letter-spacing: 0.04em;
            text-transform: uppercase;
          }
          .pred-meta { color: #94a3b8; font-size: 0.85rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _ensure_defaults()
    health = _get_json("/health", timeout=2)
    _render_sidebar(health)

    st.title("Income prediction")
    st.write(
        "Enter an applicant's details and score them with the census income "
        "model. The prediction service returns whether annual income is over "
        "50,000 dollars or at or under that line, plus the model's probability."
    )
    st.caption(
        "Trained on the 1994 US Census Adult dataset. The demographic "
        "categories are the ones the model saw in training. This demo is "
        "for the MLOps stack, not for decisions about real people."
    )

    preset_name = st.selectbox(
        "Start from an example",
        options=list(PRESET_LABELS),
        format_func=lambda key: PRESET_LABELS[key],
        key="preset_name",
    )
    if st.button("Load example"):
        _load_preset(preset_name)
        st.rerun()

    with st.form("applicant_form"):
        left, right = st.columns(2)
        with left:
            _numeric_input("age", "Age")
            _category_input("education", "Education")
            _category_input("occupation", "Occupation")
            _category_input("workClass", "Work class")
            _numeric_input("hoursPerWeek", "Hours per week")
            _numeric_input("financialWeight", "Census weight (fnlwgt)")
        with right:
            _category_input("maritalStatus", "Marital status")
            _category_input("relationship", "Relationship")
            _category_input("race", "Race")
            _category_input("sex", "Sex")
            _numeric_input("capitalGain", "Capital gain ($)")
            _numeric_input("capitalLoss", "Capital loss ($)")
        submitted = st.form_submit_button("Predict income", type="primary")

    if submitted:
        applicant = _collect_applicant()
        try:
            payload = encode_applicant(applicant)
        except ValueError as exc:
            st.error(str(exc))
        else:
            try:
                response = requests.post(
                    f"{PREDICTION_SERVICE_URL}/predict",
                    json=payload,
                    timeout=10,
                )
            except requests.RequestException:
                st.error(
                    "Could not reach the prediction service at "
                    f"{PREDICTION_SERVICE_URL}. Start it with Docker Compose "
                    "and refresh this page."
                )
            else:
                if response.ok:
                    _render_result(response.json())
                else:
                    detail = response.text
                    try:
                        detail = response.json().get("error", detail)
                    except ValueError:
                        pass
                    st.error(
                        f"The prediction service rejected this applicant. {detail}"
                    )

    _render_history()


if __name__ == "__main__":
    main()

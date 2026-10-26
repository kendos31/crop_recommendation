from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Project layout: app/app.py, models/crop_recommendation_rf.pkl,
# models/crop_recommendation_metadata.json
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "crop_recommendation_rf.pkl"
METADATA_PATH = ROOT / "models" / "crop_recommendation_metadata.json"

DEFAULT_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
DISPLAY_NAMES = {
    "N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)",
    "temperature": "Temperature (°C)", "humidity": "Relative humidity (%)",
    "ph": "Soil pH", "rainfall": "Rainfall (mm)"
}
HELP_TEXT = {
    "N": "Nitrogen value recorded for the soil sample.",
    "P": "Phosphorus value recorded for the soil sample.",
    "K": "Potassium value recorded for the soil sample.",
    "temperature": "Environmental temperature in degrees Celsius.",
    "humidity": "Relative humidity percentage.",
    "ph": "Measured soil pH.",
    "rainfall": "Rainfall measurement in millimetres."
}

crop_classes = [
    "apple", "banana", "blackgram", "chickpea", "coconut", "coffee",
    "cotton", "grapes", "jute", "kidneybeans", "lentil", "maize",
    "mango", "mothbeans", "mungbean", "muskmelon", "orange", "papaya",
    "pigeonpeas", "pomegranate", "rice", "watermelon"
]

st.set_page_config(page_title="Crop Recommendation System", page_icon="21.png", layout="wide")

#custom css
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        * {
            font-family: 'Inter', sans-serif;
        }
        
        div.stFormSubmitButton > button {
        background-color: #198754;
        color: white;
        width: 100%;
        border-radius: 10px;
        border: none;
        font-size: 18px;
        font-weight: 600;
        padding: 0.75rem;
        }

        div.stFormSubmitButton > button:hover {
        background-color: #146c43;
        color: white;
        }

        div.stFormSubmitButton > button[kind="secondary"] {
        background-color: #6c757d;
        color: white;
        }

        div.stFormSubmitButton > button[kind="secondary"]:hover {
        background-color: #5a6268;
        color: white;
        }
    </style>
    
    """,unsafe_allow_html=True)

@st.cache_resource
def load_artifacts(model_path: str, metadata_path: str):
    model = joblib.load(model_path)
    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    return model, metadata

def parse_finite_number(raw_value: str, feature: str) -> float:
    """Convert a text field to a finite float or raise a useful ValueError."""
    value_text = raw_value.strip()
    if not value_text:
        raise ValueError(f"{DISPLAY_NAMES[feature]} is required.")
    try:
        value = float(value_text)
    except (TypeError, ValueError):
        raise ValueError(f"{DISPLAY_NAMES[feature]} must be a valid number.") from None
    if not math.isfinite(value):
        raise ValueError(f"{DISPLAY_NAMES[feature]} must be finite (not NaN or infinity).")
    return value

def display_model_info():
    with st.expander("Crop classes available for prediction"):
        st.caption(
            # f" ({len(crop_classes)}): "
            ", ".join(crop_classes)
            )
        
with st.sidebar:
    st.image("21.png", width="content")
    st.title("AI-Based Crop Recommendation")
    display_model_info()
    
# st.write(
#     "Enter soil nutrient and environmental measurements to receive a crop recommendation "
#     "from the trained Random Forest classifier."
# )

st.subheader(
    "Enter soil nutrient and environmental measurements to receive a crop recommendation "
    "from the trained Random Forest classifier."
)
st.caption("Decision-support prototype only. Confirm recommendations with local agricultural expertise.")

missing_files = [p for p in (MODEL_PATH, METADATA_PATH) if not p.is_file()]
if missing_files:
    st.error("Required model artifacts were not found:")
    for path in missing_files:
        st.code(str(path))
    st.info(
        "Run the model-saving cell at the end of the Module 5 notebook, then copy "
        "`crop_recommendation_rf.pkl` and `crop_recommendation_metadata.json` into the "
        "project's `models/` directory."
    )
    st.stop()

try:
    model, metadata = load_artifacts(str(MODEL_PATH), str(METADATA_PATH))
except Exception as exc:
    st.error(f"Could not load the saved model or metadata: {exc}")
    st.stop()

features = metadata.get("features", DEFAULT_FEATURES)
if features != DEFAULT_FEATURES:
    st.warning(
        "The metadata feature order differs from the expected project order. "
        "The application will use the feature order recorded in metadata."
    )

ranges = metadata.get("training_range", {})
if not all(feature in features for feature in DEFAULT_FEATURES):
    st.error("Model metadata does not contain all seven expected input features.")
    st.stop()

defaults = {
    "N": "90", "P": "42", "K": "43", "temperature": "20.9",
    "humidity": "82.0", "ph": "6.5", "rainfall": "202.9"
}

# Apply reset values before the corresponding widgets are instantiated.
if st.session_state.pop("reset_crop_form", False):
    for feature in DEFAULT_FEATURES:
        st.session_state[f"input_{feature}"] = defaults[feature]

# Initialize input widget state once, before creating the widgets.
for feature in DEFAULT_FEATURES:
    st.session_state.setdefault(f"input_{feature}", defaults[feature])

with st.form("crop_inputs"):
    st.subheader("Soil and environmental inputs")
    st.write("Enter the seven values used by the model.")
    columns = st.columns(2)
    raw_values = {}
    for idx, feature in enumerate(DEFAULT_FEATURES):
        with columns[idx % 2]:
            raw_values[feature] = st.text_input(
                DISPLAY_NAMES[feature],
                help=HELP_TEXT[feature],
                key=f"input_{feature}"
            )
    form_col1, form_col2 = st.columns(2)
    with form_col1:
        submitted = st.form_submit_button(
            "Recommend crop", width="stretch"
        )
    with form_col2:
        clear_clicked = st.form_submit_button(
            "Clear prediction & reset",  width="stretch"
        )

# Clear/reset all input fields and saved prediction results.
if clear_clicked:
    # Streamlit does not allow changing widget state after those widgets are instantiated.
    # Use a separate reset flag; the next run applies defaults before creating widgets.
    st.session_state["reset_crop_form"] = True
    st.session_state.pop("crop_prediction_result", None)
    st.session_state.pop("crop_prediction_error", None)
    st.rerun()

if submitted:
    errors = []
    values = {}
    for feature in DEFAULT_FEATURES:
        try:
            values[feature] = parse_finite_number(raw_values.get(feature, ""), feature)
        except ValueError as exc:
            errors.append(str(exc))

    if errors:
        st.error("Please correct the following input issue(s):")
        for error in errors:
            st.write(f"- {error}")
    else:
        warnings = []
        for feature, value in values.items():
            feature_range = ranges.get(feature)
            if isinstance(feature_range, dict) and "min" in feature_range and "max" in feature_range:
                minimum, maximum = float(feature_range["min"]), float(feature_range["max"])
                if value < minimum or value > maximum:
                    warnings.append(
                        f"**{DISPLAY_NAMES[feature]} = {value:g}** is outside the observed "
                        f"training range ({minimum:g} to {maximum:g}). The prediction may be less reliable."
                    )
            else:
                warnings.append(
                    f"No observed training range is available for {DISPLAY_NAMES[feature]}; "
                    "range checking was skipped for this feature."
                )

        if warnings:
            st.warning("Input range warning")
            for warning in warnings:
                st.markdown(f"- {warning}")

        try:
            input_frame = pd.DataFrame(
                [[values[feature] for feature in features]],
                columns=features
            )
            with st.spinner("Analyzing soil and climate data... Please wait."):
                prediction = str(model.predict(input_frame)[0])
                probabilities = model.predict_proba(input_frame)[0]
                classes = [str(c) for c in model.classes_]
                if len(classes) != len(probabilities):
                    raise ValueError("Model classes and probability output have different lengths.")

                probability_table = pd.DataFrame({
                    "Crop": classes,
                    "Probability": np.asarray(probabilities, dtype=float)
                }).sort_values("Probability", ascending=False).head(3)
                confidence = float(np.max(probabilities))

            st.session_state["crop_prediction_result"] = {
                "prediction": prediction,
                "confidence": confidence,
                "probability_table": probability_table.to_dict(orient="records"),
            }
            st.session_state.pop("crop_prediction_error", None)
        except Exception as exc:
            st.session_state.pop("crop_prediction_result", None)
            st.session_state["crop_prediction_error"] = (
                f"Prediction could not be completed safely: {exc}"
            )

# Render the saved result on every rerun until the user clears/resets it.
result = st.session_state.get("crop_prediction_result")
if result:
    st.divider()
    st.subheader("Recommended crop")
    st.success(result["prediction"].title())
    st.metric("Model confidence", f'{result["confidence"]:.1%}')
    st.caption(
        "Confidence is the model's highest predicted class probability; it is not a "
        "guarantee of crop success or a calibrated measure of real-world certainty."
    )
    with st.expander("Top 3 crop probabilities"):
        probability_table = pd.DataFrame(result["probability_table"])
        probability_table["Probability"] = probability_table["Probability"].map(
            lambda value: f"{value:.1%}"
        )
        st.dataframe(probability_table,  width="stretch", hide_index=True)

prediction_error = st.session_state.get("crop_prediction_error")
if prediction_error:
    st.error(prediction_error)

with st.expander("About this prototype"):
    st.markdown(
        """
        - **Model:** saved Random Forest classifier produced by the Module 5 notebook.
        - **Inputs:** N, P, K, temperature, humidity, pH, and rainfall.
        - **Validation:** required values must be numeric and finite; observed training
          ranges are checked using the saved metadata.
        - **Output:** predicted crop, maximum class probability, and top five class probabilities.
        - **Limitations:** this dataset does not represent every location, season, soil type,
          or agronomic factor. This tool should support—not replace—professional local advice.
        """
    )

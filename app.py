
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

FEATURES = [
    "Glucose",
    "BMI",
    "Age",
    "Pregnancies",
    "DiabetesPedigreeFunction"
]
TARGET = "Outcome"

st.title("🩺 Diabetes Prediction Dashboard")
st.write("Predictive modelling using Logistic Regression and Random Forest.")

st.warning(
    "Educational use only. This model is not a medical diagnostic tool."
)

# Load dataset
DATA_PATH = Path(__file__).resolve().parent / "diabetes.csv"

if not DATA_PATH.exists():
    st.error("Please place diabetes.csv in the same folder as app.py.")
    st.stop()

@st.cache_data
def load_data(path, modified_time):
    data = pd.read_csv(path)

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    data = data.copy()

    for col in required:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["Glucose"] = data["Glucose"].replace(0, np.nan)
    data["BMI"] = data["BMI"].replace(0, np.nan)

    data = data.dropna(subset=[TARGET])
    data = data[data[TARGET].isin([0, 1])]

    for col in FEATURES:
        median = data[col].median()
        if pd.isna(median):
            raise ValueError(f"No valid values found for {col}.")
        data[col] = data[col].fillna(median)

    data[TARGET] = data[TARGET].astype(int)

    if data.empty or data[TARGET].nunique() < 2:
        raise ValueError("Dataset must contain both Outcome classes.")

    return data

try:
    df = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Dataset error: {exc}")
    st.stop()

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            random_state=42,
            class_weight="balanced",
            max_iter=1000
        ))
    ]),
    "Random Forest": RandomForestClassifier(
        random_state=42,
        n_estimators=200,
        class_weight="balanced"
    )
}

predictions = {}
metrics = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred

    metrics[name] = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(
            y_test, pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, pred, zero_division=0
        ),
        "F1-score": f1_score(
            y_test, pred, zero_division=0
        )
    }

page = st.sidebar.radio(
    "Navigation",
    ["Prediction", "Model Performance", "Dataset Explorer"]
)

if page == "Prediction":
    st.subheader("Enter patient measurements")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            glucose = st.number_input(
                "Glucose (mg/dL)", 1.0, 500.0, 120.0
            )
            bmi = st.number_input(
                "BMI", 1.0, 100.0, 28.0
            )
            age = st.number_input(
                "Age", 1, 120, 35
            )

        with col2:
            pregnancies = st.number_input(
                "Pregnancies", 0, 30, 1
            )
            pedigree = st.number_input(
                "Diabetes Pedigree Function",
                0.0, 3.0, 0.5, step=0.01
            )

        selected_model = st.selectbox(
            "Select model",
            list(models.keys()),
            index=1
        )

        submitted = st.form_submit_button(
            "Predict Outcome",
            type="primary"
        )

    if submitted:
        input_data = pd.DataFrame([{
            "Glucose": glucose,
            "BMI": bmi,
            "Age": age,
            "Pregnancies": pregnancies,
            "DiabetesPedigreeFunction": pedigree
        }], columns=FEATURES)

        model = models[selected_model]
        prediction = int(model.predict(input_data)[0])

        if prediction == 1:
            st.error("Model output: Outcome 1")
        else:
            st.success("Model output: Outcome 0")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            positive_index = list(model.classes_).index(1)
            st.metric(
                "Model-estimated probability for Outcome 1",
                f"{probabilities[positive_index]:.1%}"
            )

        st.caption(
            "This is a model estimate, not a confirmed diagnosis. "
            "Consult a qualified healthcare professional."
        )

elif page == "Model Performance":
    st.subheader("Model Evaluation")

    results = pd.DataFrame(metrics).T
    st.dataframe(
        results.style.format("{:.3f}"),
        use_container_width=True
    )

    st.subheader("Compare Models")
    st.bar_chart(results)

    selected_model = st.selectbox(
        "Confusion matrix",
        list(models.keys())
    )

    cm = confusion_matrix(
        y_test,
        predictions[selected_model],
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
    plt.close(fig)

    report = classification_report(
        y_test,
        predictions[selected_model],
        labels=[0, 1],
        target_names=["Outcome 0", "Outcome 1"],
        output_dict=True,
        zero_division=0
    )
    st.subheader("Classification Report")
    st.dataframe(
        pd.DataFrame(report).T.round(3),
        use_container_width=True
    )

    if selected_model == "Random Forest":
        importance = pd.Series(
            models[selected_model].feature_importances_,
            index=FEATURES
        ).sort_values(ascending=False)

        st.subheader("Feature Importance")
        st.bar_chart(importance)

else:
    st.subheader("Dataset Explorer")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total records", len(df))
    col2.metric("Selected features", len(FEATURES))
    col3.metric("Duplicate rows", int(df.duplicated().sum()))

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Outcome Distribution")
    st.bar_chart(df[TARGET].value_counts().sort_index())

    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        df[FEATURES + [TARGET]].corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Descriptive Statistics")
    st.dataframe(
        df[FEATURES + [TARGET]].describe().T,
        use_container_width=True
    )

st.markdown("---")
st.caption("Diabetes Prediction Dashboard | Educational project")

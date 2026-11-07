import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).resolve().parent.parent / "wine_model"
LABELS = {
    "fixed_acidity": "Фиксированная кислотность",
    "volatile_acidity": "Летучая кислотность",
    "citric_acid": "Лимонная кислота",
    "residual_sugar": "Остаточный сахар",
    "chlorides": "Хлориды",
    "free_sulfur_dioxide": "Свободный диоксид серы",
    "total_sulfur_dioxide": "Общий диоксид серы",
    "density": "Плотность",
    "pH": "pH",
    "sulphates": "Сульфаты",
    "alcohol": "Содержание алкоголя",
}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "model.joblib")


@st.cache_data
def load_test_data():
    return pd.read_csv(MODEL_DIR / "test.csv")


st.set_page_config(page_title="Оценка вина")
st.title("Оценка качества вина")

model = load_model()
test = load_test_data()
config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))

threshold = st.sidebar.slider(
    "Порог решения",
    min_value=0.0,
    max_value=1.0,
    value=float(config["threshold"]),
    step=0.001,
    format="%.3f",
)
st.sidebar.caption("Вино считается хорошим, если вероятность равна порогу или выше него.")

single_wine, = st.tabs(["Оценка одного вина"])
with single_wine:
    values = {}
    columns = st.columns(2)
    for index, (feature, label) in enumerate(LABELS.items()):
        data = test[feature]
        values[feature] = columns[index % 2].number_input(
            label,
            min_value=float(data.min()),
            max_value=float(data.max()),
            value=float(data.median()),
            step=0.00001 if feature == "density" else 0.001,
            format="%.5f" if feature == "density" else "%.3f",
        )

    color = columns[1].radio(
        "Цвет вина",
        ["Красное", "Белое"],
        index=0 if test["is_red"].mode().iat[0] == 1 else 1,
        horizontal=True,
    )
    values["is_red"] = int(color == "Красное")

    wine = pd.DataFrame([values], columns=model.feature_names_in_)
    good_class = list(model.classes_).index(1)
    probability = model.predict_proba(wine)[0, good_class]

    st.metric("Вероятность, что вино хорошее", f"{probability:.2%}")
    if probability >= threshold:
        st.success("Решение модели: хорошее вино")
    else:
        st.warning("Решение модели: вино не прошло порог")

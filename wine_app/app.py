import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)

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

good_class = list(model.classes_).index(1)
single_wine, model_metrics = st.tabs(["Оценка одного вина", "Модельные метрики"])
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


with model_metrics:
    actual = test["target"]
    scores = model.predict_proba(test[model.feature_names_in_])[:, good_class]
    predicted = scores >= threshold
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, average="binary", zero_division=0
    )

    st.subheader("Матрица ошибок")
    st.table(pd.DataFrame(
        matrix,
        index=["Факт ° класс 0", "Факт ° класс 1"],
        columns=["Прогноз ° класс 0", "Прогноз ° класс 1"],
    ))
    st.caption("Класс 1 - хорошее вино. Класс 0 - остальные вина.")

    precision_column, recall_column, f1_column = st.columns(3)
    precision_column.metric("Precision", f"{precision:.3f}" if tp + fp else "-")
    recall_column.metric("Recall", f"{recall:.3f}")
    f1_column.metric("F₁", f"{f1:.3f}")

    false_positive_rates, true_positive_rates, _ = roc_curve(
        actual, scores, drop_intermediate=False
    )
    precisions, recalls, _ = precision_recall_curve(actual, scores)
    figure = Figure(figsize=(9, 4), layout="constrained")
    roc_axis, pr_axis = figure.subplots(1, 2)

    roc_axis.plot(false_positive_rates, true_positive_rates)
    roc_axis.scatter(fp / (fp + tn), recall, color="tab:red", zorder=3)
    roc_axis.set(title="ROC", xlabel="False positive rate", ylabel="Recall")

    pr_axis.plot(recalls, precisions)
    pr_axis.scatter(recall, precision if tp + fp else 1, color="tab:red", zorder=3)
    pr_axis.set(title="PR", xlabel="Recall", ylabel="Precision")

    st.pyplot(figure)
    st.caption(f"Красная точка показывает текущий порог {threshold:.3f}.")
    if not (tp + fp):
        st.caption(
            "В подборке нет вин, поэтому precision не определён. "
            "На PR-кривой этому случаю соответствует условная точка (0, 1)."
        )

    st.write(
        f"При текущем пороге в подборку попадают {tp + fp} из {len(test)} вин. "
        "Если поднять порог, отбор станет строже. В подборку может попасть меньше "
        "вин, и часть хороших вин может быть пропущена. Если снизить порог, "
        "в подборку может попасть больше хороших вин, а вместе с ними и больше "
        "вин, ошибочно признанных хорошими."
    )

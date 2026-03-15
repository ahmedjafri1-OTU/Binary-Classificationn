import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay


def main():
    st.title("Binary Classification App")
    st.sidebar.title("Binary Classification App")
    st.markdown(
        "Are the mushrooms edible or poisonous? This app uses machine learning "
        "to predict whether a mushroom is edible or poisonous based on its features."
    )

    @st.cache_data
    def load_data():
        data = pd.read_csv("mushrooms.csv")
        label_encoder = LabelEncoder()
        for column in data.columns:
            data[column] = label_encoder.fit_transform(data[column])
        return data

    @st.cache_data
    def split(df):
        X = df.drop("type", axis=1)
        y = df["type"]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    df = load_data()
    X_train, X_test, y_train, y_test = split(df)

    if st.sidebar.checkbox("Show raw data"):
        st.subheader("Mushroom Dataset")
        st.write(df)

    classifier = st.sidebar.selectbox(
        "Choose Classifier",
        ("Logistic Regression", "Random Forest", "SVM")
    )

    # Hyperparameter options
    if classifier == "Logistic Regression":
        st.sidebar.subheader("Logistic Regression Hyperparameters")
        C = st.sidebar.number_input("C", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        max_iter = st.sidebar.slider("max_iter", 100, 2000, 1000, 100)
        model = LogisticRegression(C=C, max_iter=max_iter)

    elif classifier == "Random Forest":
        st.sidebar.subheader("Random Forest Hyperparameters")
        n_estimators = st.sidebar.slider("n_estimators", 10, 500, 100, 10)
        max_depth = st.sidebar.slider("max_depth", 1, 50, 10, 1)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

    else:
        st.sidebar.subheader("SVM Hyperparameters")
        C = st.sidebar.number_input("C", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
        kernel = st.sidebar.selectbox("kernel", ("linear", "rbf", "poly", "sigmoid"))
        model = SVC(C=C, kernel=kernel, probability=True)

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    st.subheader("Model Accuracy")
    st.write(accuracy)

    metrics = st.sidebar.multiselect(
        "What metrics to plot?",
        ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve")
    )

    if "Confusion Matrix" in metrics:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
        st.pyplot(fig)

    if "ROC Curve" in metrics:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots()
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
        st.pyplot(fig)

    if "Precision-Recall Curve" in metrics:
        st.subheader("Precision-Recall Curve")
        fig, ax = plt.subplots()
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax)
        st.pyplot(fig)


if __name__ == "__main__":
    main()
"""
train.py
--------
End-to-end training pipeline for the Spam Email/SMS Classifier.

Steps:
1. Load dataset (data/spam.csv)
2. Clean text (preprocess.py)
3. TF-IDF feature extraction
4. Train multiple models (Naive Bayes, Logistic Regression, SVM)
5. Evaluate each on a held-out test set
6. Save the best model + vectorizer for the Streamlit app
7. Save evaluation plots (confusion matrix, model comparison)
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from preprocess import clean_text

DATA_PATH = "data/spam.csv"
MODEL_DIR = "model"
OUT_DIR = "outputs"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["label", "message"])
    df = df.drop_duplicates(subset=["message"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])]
    return df


def main():
    print("1) Loading data...")
    df = load_data()
    print(f"   Total messages: {len(df)}  (spam={sum(df.label=='spam')}, ham={sum(df.label=='ham')})")

    print("2) Cleaning text...")
    df["clean_message"] = df["message"].apply(clean_text)

    # encode labels: spam=1, ham=0
    df["target"] = (df["label"] == "spam").astype(int)

    print("3) Splitting train/test (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_message"], df["target"],
        test_size=0.2, random_state=42, stratify=df["target"]
    )

    print("4) TF-IDF vectorization...")
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("5) Training models...")
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
    }

    results = []
    best_model_name, best_model, best_f1 = None, None, -1

    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})
        print(f"   {name:22s} | Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    print(f"\n6) Best model: {best_model_name} (F1={best_f1:.3f})")

    # Detailed report for best model
    best_preds = best_model.predict(X_test_vec)
    print("\nClassification report (best model):")
    print(classification_report(y_test, best_preds, target_names=["ham", "spam"]))

    # Save confusion matrix plot
    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    # Save model comparison bar chart
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)

    ax = results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]].plot(
        kind="bar", figsize=(8, 5), ylim=(0, 1.05)
    )
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/model_comparison.png", dpi=150)
    plt.close()

    # Save best model + vectorizer for the app
    joblib.dump(best_model, f"{MODEL_DIR}/spam_model.pkl")
    joblib.dump(vectorizer, f"{MODEL_DIR}/vectorizer.pkl")
    with open(f"{MODEL_DIR}/best_model_name.txt", "w") as f:
        f.write(best_model_name)

    print(f"\nSaved model -> {MODEL_DIR}/spam_model.pkl")
    print(f"Saved vectorizer -> {MODEL_DIR}/vectorizer.pkl")
    print(f"Saved plots -> {OUT_DIR}/confusion_matrix.png , {OUT_DIR}/model_comparison.png")


if __name__ == "__main__":
    main()

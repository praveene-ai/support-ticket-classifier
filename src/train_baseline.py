from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline


def explain_confusion(
    text: str, target_type: str, true_label: str, pred_label: str, runner_up: str, is_error: bool
) -> str:
    """Provide a targeted explanation for misclassified or low-confidence tickets."""
    competing = pred_label if is_error else runner_up
    tokens = text.lower().split()

    if not is_error:
        if "thanks!" in tokens or "hello," in tokens or "hi" in tokens:
            return f"Polite greeting/sign-off dilutes TF-IDF weight, narrowing margin against '{competing}'."
        if "urgent:" in tokens or "asap" in tokens:
            return f"Urgency modifier ('urgent'/'asap') shifts probability toward '{competing}' despite base template."
        return f"Shared n-gram vocabulary with '{competing}' reduces confidence margin under L2 regularization."

    if len(tokens) <= 5:
        return "Very short ticket text provides sparse TF-IDF n-gram signal."
    return f"Overlapping keywords caused confusion between '{true_label}' and '{competing}'."


def tune_and_evaluate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_col: str,
) -> tuple[Pipeline, pd.DataFrame]:
    """Build, cross-validate, evaluate, and analyze errors for a TF-IDF + LogisticRegression pipeline."""
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "clf__C": [0.1, 1.0, 5.0, 10.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(train_df["text"], train_df[target_col])

    best_model: Pipeline = search.best_estimator_
    y_true = val_df[target_col].to_numpy()
    y_pred = best_model.predict(val_df["text"])
    y_proba = best_model.predict_proba(val_df["text"])
    classes = list(best_model.classes_)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    print(f"\n{'=' * 70}")
    print(f"TARGET: {target_col.upper()}")
    print(f"{'=' * 70}")
    print(f"Best CV Params : {search.best_params_} (CV Macro F1: {search.best_score_:.4f})")
    print(f"Val Accuracy   : {acc:.4f}")
    print(f"Val Macro F1   : {macro_f1:.4f}")
    print("\nConfusion Matrix (rows = True, cols = Predicted):")
    cm_df = pd.DataFrame(cm, index=[f"True:{c}" for c in classes], columns=[f"Pred:{c}" for c in classes])
    print(cm_df.to_string())

    class_to_idx = {c: i for i, c in enumerate(classes)}
    true_probs = np.array([y_proba[i, class_to_idx[y_true[i]]] for i in range(len(y_true))])

    sorted_idx = np.argsort(y_proba, axis=1)
    top1_prob = y_proba[np.arange(len(y_true)), sorted_idx[:, -1]]
    top2_prob = y_proba[np.arange(len(y_true)), sorted_idx[:, -2]]
    runner_up_labels = [classes[idx] for idx in sorted_idx[:, -2]]
    margin = top1_prob - top2_prob
    is_error = (y_true != y_pred).astype(float)

    confusion_score = is_error * (1.0 + (top1_prob - true_probs)) + (1.0 - margin) * 0.5

    analysis_df = pd.DataFrame(
        {
            "target": target_col,
            "text": val_df["text"].values,
            "true_label": y_true,
            "pred_label": y_pred,
            "runner_up": runner_up_labels,
            "true_prob": np.round(true_probs, 3),
            "pred_prob": np.round(top1_prob, 3),
            "margin": np.round(margin, 3),
            "is_error": y_true != y_pred,
            "confusion_score": confusion_score,
        }
    )
    analysis_df["guess_why"] = [
        explain_confusion(row.text, target_col, row.true_label, row.pred_label, row.runner_up, row.is_error)
        for row in analysis_df.itertuples()
    ]

    return best_model, analysis_df


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")

    cat_model, cat_errors = tune_and_evaluate(train_df, val_df, "category")
    prio_model, prio_errors = tune_and_evaluate(train_df, val_df, "priority")

    cat_acc = accuracy_score(val_df["category"], cat_model.predict(val_df["text"]))
    cat_f1 = f1_score(val_df["category"], cat_model.predict(val_df["text"]), average="macro")
    prio_acc = accuracy_score(val_df["priority"], prio_model.predict(val_df["text"]))
    prio_f1 = f1_score(val_df["priority"], prio_model.predict(val_df["text"]), average="macro")

    print(f"\n{'=' * 70}")
    print(
        f"SUMMARY -> Category: accuracy {cat_acc:.2f}, macro F1 {cat_f1:.2f} | "
        f"Priority: accuracy {prio_acc:.2f}, macro F1 {prio_f1:.2f}"
    )
    print(f"{'=' * 70}")

    combined_confusing = (
        pd.concat([cat_errors, prio_errors], ignore_index=True)
        .sort_values(by="confusion_score", ascending=False)
        .head(10)
    )

    print("\nTOP 10 MOST CONFUSING TICKET EXAMPLES (VALIDATION SET):")
    print("-" * 70)
    for rank, row in enumerate(combined_confusing.itertuples(), start=1):
        status = "MISCLASSIFIED" if row.is_error else f"LOW-MARGIN vs {row.runner_up}"
        print(
            f"{rank:2d}. [{row.target.upper()} | {status}] "
            f"True: {row.true_label} (p={row.true_prob:.2f}) -> "
            f"Pred: {row.pred_label} (p={row.pred_prob:.2f}, margin={row.margin:.2f})"
        )
        print(f"    Text : \"{row.text}\"")
        print(f"    Why? : {row.guess_why}")

    bundle_path = models_dir / "baseline_tfidf_logreg.joblib"
    joblib.dump(
        {
            "category_pipeline": cat_model,
            "priority_pipeline": prio_model,
        },
        bundle_path,
    )
    print(f"\nSaved baseline models to: {bundle_path}")


if __name__ == "__main__":
    main()
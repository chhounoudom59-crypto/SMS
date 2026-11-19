"""Train leak-free TF-IDF pipelines and compare three SMS classifiers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data_loader import load_dataset

ROOT = Path(__file__).resolve().parent
MODEL_NAMES = {
    'naive_bayes': 'Multinomial Naive Bayes',
    'logistic_regression': 'Logistic Regression',
    'linear_svm': 'Linear SVM',
}


def build_models() -> dict[str, Pipeline]:
    """Each classifier owns its own training-only TF-IDF transformer."""
    estimators = {
        'naive_bayes': MultinomialNB(alpha=1.0),
        'logistic_regression': LogisticRegression(max_iter=2000, random_state=42),
        'linear_svm': LinearSVC(random_state=42),
    }
    return {
        name: Pipeline([
            ('tfidf', TfidfVectorizer(lowercase=True, strip_accents='unicode',
                                     ngram_range=(1, 2), min_df=2,
                                     sublinear_tf=True)),
            ('classifier', clf),
        ])
        for name, clf in estimators.items()
    }


def evaluate(y_true, y_pred) -> dict:
    """Spam is the positive class; confusion matrix is ordered ham, spam."""
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'spam_precision': float(precision_score(y_true, y_pred, pos_label='spam', zero_division=0)),
        'spam_recall': float(recall_score(y_true, y_pred, pos_label='spam', zero_division=0)),
        'spam_f1': float(f1_score(y_true, y_pred, pos_label='spam', zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=['ham', 'spam']).tolist(),
        'classification_report': classification_report(y_true, y_pred,
                                      labels=['ham', 'spam'], zero_division=0,
                                      output_dict=True),
    }


def train_and_save(dataset_path: Path | None = None, output_dir: Path = ROOT) -> pd.DataFrame:
    df, cleaning = load_dataset(dataset_path)
    # The single untouched holdout is only evaluated once, after fixed model selection.
    x_train, x_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.20, stratify=df['label'], random_state=42
    )
    models_dir = output_dir / 'models'
    reports_dir = output_dir / 'reports'
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    results = []
    complete_reports = {}
    for name, model in build_models().items():
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        scores = evaluate(y_test, preds)
        joblib.dump(model, models_dir / f'{name}.joblib')
        results.append({'model': name, 'display_name': MODEL_NAMES[name],
                        **{key: scores[key] for key in (
                            'accuracy', 'spam_precision', 'spam_recall', 'spam_f1', 'macro_f1')}})
        complete_reports[name] = scores
        print(f"{MODEL_NAMES[name]}: spam F1={scores['spam_f1']:.4f}, "
              f"precision={scores['spam_precision']:.4f}, recall={scores['spam_recall']:.4f}")

    comparison = pd.DataFrame(results).sort_values('spam_f1', ascending=False)
    comparison.to_csv(reports_dir / 'model_comparison.csv', index=False)
    (reports_dir / 'evaluation.json').write_text(
        json.dumps({'dataset': cleaning, 'train_rows': len(x_train),
                    'test_rows': len(x_test), 'seed': 42,
                    'model_scores': complete_reports,
                    'warning': 'Scores are from a single held-out split; not cross-validated.'}, indent=2),
        encoding='utf-8')
    print('\nComparison (held-out test set):\n', comparison.to_string(index=False))
    return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate UCI SMS spam classifiers')
    parser.add_argument('--dataset', type=Path, default=None,
                        help='Optional path to extracted SMSSpamCollection TSV')
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    train_and_save(args.dataset, args.output_dir)

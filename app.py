"""Streamlit interface for the Chapter 4 SMS Spam Classification demo."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_NAMES = {
    'naive_bayes': 'Naive Bayes',
    'logistic_regression': 'Logistic Regression',
    'linear_svm': 'Linear SVM',
}

st.set_page_config(page_title='Chapter 4 · SMS Spam Detection', page_icon='📩', layout='wide')

@st.cache_resource
def load_models() -> dict:
    return {name: joblib.load(ROOT / 'models' / f'{name}.joblib')
            for name in MODEL_NAMES}

st.title('📩 SMS Spam Classification Lab')
st.caption('Chapter 4: Text Classification · TF-IDF + Naive Bayes / Logistic Regression / Linear SVM')
st.write('Type an SMS message and compare the predictions of three trained classifiers.')

missing = [n for n in MODEL_NAMES if not (ROOT / 'models' / f'{n}.joblib').exists()]
if missing:
    st.error('Trained models not found. Run `python train.py` in the project directory first.')
    st.stop()

models = load_models()
example = 'Congratulations! You have won a free prize. Claim it now!'
message = st.text_area('SMS message', value=example, height=130, max_chars=5000)

if st.button('Classify message', type='primary', use_container_width=True):
    if not message.strip():
        st.warning('Please enter a non-empty message.')
    else:
        columns = st.columns(3)
        for column, (name, model) in zip(columns, models.items()):
            prediction = str(model.predict([message])[0])
            with column:
                st.subheader(MODEL_NAMES[name])
                if prediction == 'spam':
                    st.error('SPAM')
                else:
                    st.success('HAM (not spam)')
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba([message])[0]
                    spam_index = list(model.classes_).index('spam')
                    st.metric('Estimated P(spam)', f'{probabilities[spam_index]:.1%}')
                else:
                    margin = float(model.decision_function([message])[0])
                    spam_index = list(model.classes_).index('spam')
                    if spam_index == 0:
                        margin = -margin
                    st.metric('SVM decision margin (spam-positive)', f'{margin:.3f}')
                    st.caption('Margin is not a probability.')

st.divider()
st.subheader('Evaluation on the held-out test split')
comparison_path = ROOT / 'reports' / 'model_comparison.csv'
report_path = ROOT / 'reports' / 'evaluation.json'
if comparison_path.exists() and report_path.exists():
    comparison = pd.read_csv(comparison_path)
    st.dataframe(comparison.rename(columns={
        'display_name': 'Model', 'accuracy': 'Accuracy',
        'spam_precision': 'Spam precision', 'spam_recall': 'Spam recall',
        'spam_f1': 'Spam F1', 'macro_f1': 'Macro F1'
    }).drop(columns=['model']).style.format(precision=3), use_container_width=True, hide_index=True)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    st.caption(f"Train samples: {report['train_rows']:,} · Test samples: {report['test_rows']:,} · "
               f"Random seed: {report['seed']}")
    st.caption('Confusion matrix order: rows = actual [ham, spam]; columns = predicted [ham, spam].')
    selection = st.selectbox('Inspect confusion matrix', list(MODEL_NAMES),
                             format_func=lambda x: MODEL_NAMES[x])
    matrix = report['model_scores'][selection]['confusion_matrix']
    st.dataframe(pd.DataFrame(matrix, index=['Actual ham', 'Actual spam'],
                              columns=['Predicted ham', 'Predicted spam']))
    with st.expander('Dataset cleaning and split metadata'):
        st.json(report['dataset'])
else:
    st.info('Run `python train.py` to generate the evaluation reports.')

st.divider()
st.caption('Educational demo only. The corpus is historical English-language SMS; do not use this model as a production security filter without further validation.')

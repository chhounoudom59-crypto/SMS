from pathlib import Path
import pandas as pd
from data_loader import load_dataset
from train import build_models, evaluate, train_and_save


def test_data_cleaning(tmp_path):
    path = tmp_path / 'SMSSpamCollection'
    path.write_text('ham\thello there\nspam\twin now\nham\thello there\nspam\thello there\nham\tsee you\nspam\tfree prize\n', encoding='utf-8')
    data, stats = load_dataset(path)
    assert stats['original_rows'] == 6
    assert stats['conflicting_rows_removed'] == 3
    assert len(data) == 3


def test_models_train_on_text():
    texts = ['free gift now', 'free money win', 'get cash prize', 'win big reward',
             'call you tomorrow', 'meeting next week', 'dinner at home', 'see you soon']
    labels = ['spam'] * 4 + ['ham'] * 4
    for model in build_models().values():
        model.fit(texts, labels)
        assert model.predict(['free money']).shape == (1,)


def test_metrics():
    m = evaluate(['spam', 'spam', 'ham', 'ham'], ['spam', 'ham', 'spam', 'ham'])
    assert m['accuracy'] == 0.5
    assert m['confusion_matrix'] == [[1, 1], [1, 1]]

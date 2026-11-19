"""Load the original UCI SMS Spam Collection safely."""
from __future__ import annotations

import io
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

URL = 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'SMSSpamCollection'


def download_dataset(destination: Path = DATA_PATH) -> Path:
    """Download the official archive, extracting only the expected data member."""
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(URL, timeout=45) as response:
            archive_bytes = response.read()
        with ZipFile(io.BytesIO(archive_bytes)) as archive:
            contents = archive.read('SMSSpamCollection')
    except Exception as exc:
        raise RuntimeError(
            'Could not download UCI dataset. Download the ZIP from ' + URL +
            ' and extract SMSSpamCollection to ' + str(destination)
        ) from exc
    destination.write_bytes(contents)
    return destination


def load_dataset(path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Validate labels, remove exact duplicate texts, and report cleaning counts."""
    source = Path(path) if path is not None else download_dataset()
    df = pd.read_csv(source, sep='\t', header=None, names=['label', 'text'],
                     encoding='utf-8', on_bad_lines='error')
    original_count = len(df)
    df = df.dropna(subset=['label', 'text']).copy()
    df['text'] = df['text'].astype(str).str.strip()
    df = df.loc[df['text'].ne('')].copy()
    if not df['label'].isin(['ham', 'spam']).all():
        raise ValueError('Dataset contains labels outside ham/spam.')
    # Ambiguous identical messages with conflicting labels are excluded, not arbitrarily chosen.
    ambiguous = df.groupby('text')['label'].nunique()
    conflicting_texts = set(ambiguous[ambiguous > 1].index)
    conflicting_rows = int(df['text'].isin(conflicting_texts).sum())
    df = df.loc[~df['text'].isin(conflicting_texts)]
    exact_duplicates = int(df.duplicated(subset=['text']).sum())
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    if df['label'].nunique() != 2:
        raise ValueError('Both ham and spam classes must be present.')
    return df, {
        'original_rows': original_count,
        'conflicting_rows_removed': conflicting_rows,
        'duplicate_rows_removed': exact_duplicates,
        'rows_after_cleaning': len(df),
        'label_counts': {str(k): int(v) for k, v in df['label'].value_counts().items()},
    }

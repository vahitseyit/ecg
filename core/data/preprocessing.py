import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from core.data.loader import PTBXLDataLoader

def load_signals_and_labels(
    data_path='data/raw/ptbxl',
    sampling_rate=500,
    max_records=None
):
    loader = PTBXLDataLoader(data_path=data_path, sampling_rate=sampling_rate)
    loader.load_metadata()
    signals = []
    labels = []
    ecg_ids = []
    n_records = len(loader.df) if max_records is None else min(max_records, len(loader.df))
    for idx in range(n_records):
        try:
            signal, label, ecg_id = loader.get_signal_and_labels(idx)
            signals.append(signal)
            labels.append(label)
            ecg_ids.append(ecg_id)
        except Exception as e:
            print(f"Hata (idx={idx}): {e}")
    signals = np.array(signals)
    return signals, labels, ecg_ids

def binarize_labels(labels):
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)
    return y, mlb

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.data.preprocessing import load_signals_and_labels, binarize_labels

signals, labels, ecg_ids = load_signals_and_labels(
    data_path='data/raw/ptbxl',
    sampling_rate=500,
    max_records=100  # Hızlı test için 100 kayıt
)

print("Sinyal dizisi şekli:", signals.shape)
print("İlk 3 etiket:", labels[:3])

y, mlb = binarize_labels(labels)
print("Binarize etiket şekli:", y.shape)
print("Tüm hastalık türleri:", mlb.classes_)

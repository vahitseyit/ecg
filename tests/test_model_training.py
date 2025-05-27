import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from core.data.preprocessing import load_signals_and_labels, binarize_labels
from core.models.train_and_predict import train_model
from sklearn.model_selection import train_test_split

signals, labels, ecg_ids = load_signals_and_labels(
    data_path='data/raw/ptbxl',
    sampling_rate=500,
    max_records=100  # Test için küçük tut
)
y, mlb = binarize_labels(labels)
X_train, X_test, y_train, y_test = train_test_split(signals, y, test_size=0.2, random_state=42)
X_train_t = torch.tensor(X_train, dtype=torch.float32).permute(0,2,1)
X_test_t = torch.tensor(X_test, dtype=torch.float32).permute(0,2,1)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

model = train_model(X_train_t, y_train_t, X_test_t, y_test_t, n_classes=y.shape[1], n_epochs=2, batch_size=8)

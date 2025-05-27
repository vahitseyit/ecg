import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import torch
import numpy as np
import pickle
from core.models.train_and_predict import load_model, predict_signal
import matplotlib.pyplot as plt

def predict_from_signal(signal: np.ndarray):
    """
    EKG sinyalini alır, 12 kanala kopyalar, modele uygun hale getirir ve tahmin döner.
    Returns:
        pred_labels: Tahmin edilen etiketler
        pred_scores: Skorlar (olasılıklar)
    """
    # 2. Sinyali 12 kanala kopyala
    signal_12 = np.tile(signal, (12, 1)).T  # shape: (num_samples, 12)
    # 3. Gerekirse uzunluğu modele uygun kırp/kısalt
    target_length = 5000  # PTB-XL kayıtlarının tipik uzunluğu
    if signal_12.shape[0] > target_length:
        signal_12 = signal_12[:target_length, :]
    elif signal_12.shape[0] < target_length:
        pad_width = target_length - signal_12.shape[0]
        signal_12 = np.pad(signal_12, ((0, pad_width), (0, 0)), mode='constant')
    # 4. Modeli ve label binarizer'ı yükle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('models_artifacts/mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)
    n_classes = len(mlb.classes_)
    model = load_model('models_artifacts/best_model.pt', in_channels=12, n_classes=n_classes, device=device)
    # 5. Tahmin yap
    pred_labels, pred_scores = predict_signal(model, signal_12, mlb, device=device)
    return pred_labels, pred_scores

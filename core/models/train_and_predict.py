import torch
from torch.utils.data import DataLoader, TensorDataset
from core.models.architecture import InceptionTime1D
import numpy as np

def get_dataloaders(X_train_t, y_train_t, X_test_t, y_test_t, batch_size=32):
    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    return train_dl, test_dl

def train_model(train_loader, test_loader, n_classes, n_epochs=10, lr=1e-3, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InceptionTime1D(in_channels=12, n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss/len(train_loader):.4f}")
    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model kaydedildi: {path}")

def load_model(path, in_channels, n_classes, device=None):
    from core.models.architecture import InceptionTime1D
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InceptionTime1D(in_channels=in_channels, n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"Model yüklendi: {path}")
    return model

def predict_signal(model, signal, mlb, device=None):
    """
    signal: numpy array, shape (5000, 12) veya (12, 5000)
    mlb: MultiLabelBinarizer (etiket isimleri için)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Giriş şekli (1, 12, 5000) olmalı
    if signal.shape[0] == 5000 and signal.shape[1] == 12:
        signal = signal.T  # (12, 5000)
    signal_t = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(signal_t)
        preds = preds.cpu().numpy()[0]
    pred_labels = [mlb.classes_[i] for i, p in enumerate(preds) if p > 0.5]
    return pred_labels, preds

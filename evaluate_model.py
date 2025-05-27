import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
from core.data.ptbxl_dataset import PTBXLDataset
from core.models.train_and_predict import load_model
from core.data.preprocessing import binarize_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

# --- Ayarlar ---
DATA_PATH = 'data/raw/ptbxl'
SAMPLING_RATE = 500
MODEL_PATH = 'models_artifacts/best_model.pt'
MLB_PATH = 'models_artifacts/mlb.pkl'
BATCH_SIZE = 32

# --- MultiLabelBinarizer'ı yükle ---
with open(MLB_PATH, 'rb') as f:
    mlb = pickle.load(f)

# --- Metadata ve etiketleri yükle ---
from core.data.loader import PTBXLDataLoader
loader = PTBXLDataLoader(data_path=DATA_PATH, sampling_rate=SAMPLING_RATE)
loader.load_metadata()
labels = []
for idx in range(len(loader.df)):
    try:
        _, label, _ = loader.get_signal_and_labels(idx)
        labels.append(label)
    except Exception as e:
        print(f"Hata (idx={idx}): {e}")

# --- Multi-label binarizasyon ---
y, _ = binarize_labels(labels)

# --- Eğitim/test index bölmesi ---
indices = list(range(len(labels)))
_, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

# --- Test Dataset ve DataLoader ---
test_dataset = PTBXLDataset(DATA_PATH, SAMPLING_RATE, test_idx, label_binarizer=mlb)

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn
)

# --- Modeli yükle ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = load_model(MODEL_PATH, in_channels=12, n_classes=y.shape[1], device=device)

# --- Değerlendirme fonksiyonu ---
def evaluate_model(model, data_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                continue
            xb, yb = batch
            xb = xb.to(device)
            preds = model(xb).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(yb.numpy())
    y_true = np.concatenate(all_targets, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    y_pred_bin = (y_pred > 0.5).astype(int)

    print('F1 (macro):', f1_score(y_true, y_pred_bin, average='macro'))
    print('F1 (micro):', f1_score(y_true, y_pred_bin, average='micro'))
    print('Precision (macro):', precision_score(y_true, y_pred_bin, average='macro'))
    print('Recall (macro):', recall_score(y_true, y_pred_bin, average='macro'))
    try:
        print('ROC-AUC (macro):', roc_auc_score(y_true, y_pred, average='macro'))
    except Exception as e:
        print('ROC-AUC hesaplanamadı:', e)

# --- Modeli değerlendir ---
evaluate_model(model, test_loader, device=device) 
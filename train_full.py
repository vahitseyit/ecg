import torch
from torch.utils.data import DataLoader
from core.data.preprocessing import binarize_labels
from core.data.ptbxl_dataset import PTBXLDataset
from core.models.train_and_predict import train_model, save_model
from sklearn.model_selection import train_test_split
import pickle
import os

# --- Ayarlar ---
DATA_PATH = 'data/raw/ptbxl'
SAMPLING_RATE = 500
MODEL_PATH = 'models_artifacts/best_model.pt'
MLB_PATH = 'models_artifacts/mlb.pkl'
BATCH_SIZE = 32
N_EPOCHS = 20
LR = 1e-3

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
y, mlb = binarize_labels(labels)
print(f"Toplam etiket: {y.shape[1]}")

# --- Eğitim/test index bölmesi ---
indices = list(range(len(labels)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
print(f"Eğitim seti: {len(train_idx)}, Test seti: {len(test_idx)}")

# --- Dataset ve DataLoader oluştur ---
train_dataset = PTBXLDataset(DATA_PATH, SAMPLING_RATE, train_idx, label_binarizer=mlb)
test_dataset = PTBXLDataset(DATA_PATH, SAMPLING_RATE, test_idx, label_binarizer=mlb)

def collate_fn(batch):
    # None olanları at
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn
)

# --- Modeli eğit ---
print('Model eğitimi başlıyor...')
model = train_model(
    train_loader, test_loader,
    n_classes=y.shape[1],
    n_epochs=N_EPOCHS,
    lr=LR
)

# --- Modeli kaydet ---
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
save_model(model, MODEL_PATH)

# --- MultiLabelBinarizer'ı kaydet ---
with open(MLB_PATH, 'wb') as f:
    pickle.dump(mlb, f)
print(f"MultiLabelBinarizer kaydedildi: {MLB_PATH}")

print('Eğitim ve kayıt tamamlandı!')

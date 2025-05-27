import torch
from torch.utils.data import Dataset
from core.data.loader import PTBXLDataLoader
import random

class PTBXLDataset(Dataset):
    def __init__(self, data_path, sampling_rate, indices, label_binarizer=None):
        self.loader = PTBXLDataLoader(data_path=data_path, sampling_rate=sampling_rate)
        self.loader.load_metadata()
        self.indices = indices
        self.label_binarizer = label_binarizer

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        tries = 0
        while tries < 3:
            real_idx = self.indices[idx]
            try:
                signal, label, ecg_id = self.loader.get_signal_and_labels(real_idx)
                if self.label_binarizer:
                    label = self.label_binarizer.transform([label])[0]
                signal = torch.tensor(signal, dtype=torch.float32).permute(1, 0)  # (leads, length)
                label = torch.tensor(label, dtype=torch.float32)
                return signal, label
            except Exception as e:
                print(f"Hata (idx={real_idx}): {e}")
                # random başka bir index dene
                idx = random.randint(0, len(self.indices) - 1)
                tries += 1
        # 3 denemede de olmadıysa None döndür
        return None

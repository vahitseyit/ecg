import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.data.loader import PTBXLDataLoader

csv_path = 'data/raw/ptbxl/ptbxl_database.csv'
data_path = 'data/raw/ptbxl'

# 500 Hz için loader başlat
loader = PTBXLDataLoader(data_path=data_path, sampling_rate=500)
loader.load_metadata()

# İlk kaydı test et
signal, labels, ecg_id = loader.get_signal_and_labels(0)
print("ECG ID:", ecg_id)
print("Etiketler:", labels)
print("Sinyal şekli:", signal.shape)  # (ör: 5000, 12)
import pandas as pd
from pathlib import Path
import os
import ast
import wfdb

class PTBXLDataLoader:
    def __init__(self, data_path: str = None, sampling_rate: int = 100):
        if data_path is None:
            # Varsayılan yol
            data_path = os.path.join(os.getcwd(), "data", "raw", "ptbxl")
        self.data_path = Path(data_path)
        self.sampling_rate = sampling_rate  # 100 veya 500
        self.df = None

    def load_metadata(self):
        """Meta verileri yükle ve temel istatistikleri göster"""
        metadata_path = self.data_path / "ptbxl_database.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Meta veri dosyası bulunamadı: {metadata_path}")
        print(f"Dosya yolu: {metadata_path}")
        df = pd.read_csv(metadata_path)
        # Temel istatistikler
        print(f"Toplam kayıt sayısı: {len(df)}")
        print(f"\nYaş dağılımı:\n{df['age'].describe()}")
        print(f"\nCinsiyet dağılımı:\n{df['sex'].value_counts()}")
        self.df = df
        return df

    def get_record_info(self, idx):
        """Belirli bir indexteki kaydın sinyal dosyası yolunu ve etiketlerini döndürür."""
        if self.df is None:
            self.load_metadata()
        row = self.df.iloc[idx]
        # sampling_rate'a göre doğru klasörü seç
        folder = f"records{self.sampling_rate}"
        # filename_hr ör: records500/00000/00001_hr
        # Sadece alt yolunu alıp, ana data_path ile birleştir
        rel_path = Path(row['filename_hr'])
        # Klasör ismini güncelle (kullanıcı sampling_rate değiştirdiyse)
        rel_path = Path(folder) / rel_path.relative_to(rel_path.parts[0])
        record_path = self.data_path / rel_path
        labels = list(ast.literal_eval(row['scp_codes']).keys())
        ecg_id = row['ecg_id']
        return record_path, labels, ecg_id

    def load_signal(self, record_path):
        """Sinyal dosyasını (wfdb ile) okur ve numpy array olarak döndürür. Hata kontrolü yapar."""
        # wfdb.rdrecord uzantısız path ister
        if not (record_path.with_suffix('.hea')).exists():
            raise FileNotFoundError(f"Sinyal dosyası bulunamadı: {record_path.with_suffix('.hea')}")
        record = wfdb.rdrecord(str(record_path))
        return record.p_signal

    def get_signal_and_labels(self, idx):
        """Hem sinyali hem etiketleri ve ecg_id'yi döndürür."""
        record_path, labels, ecg_id = self.get_record_info(idx)
        signal = self.load_signal(record_path)
        return signal, labels, ecg_id

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

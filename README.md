├── core/ # Model ve yardımcı kodlar.

├── models_artifacts/ # Eğitilmiş model ağırlıkları.

├── data/ # Ham veri (PTB-XL).

├── notebooks/ # Jupyter analiz defterleri.

├── tests/ # Otomatik testler.

├── apps/ # Web uygulaması.

├── README.md.



---

## Kurulum

1. Gerekli Python paketlerini yükleyin:
    ```bash
    pip install -r requirements.txt
    ```
2. PTB-XL veri setini [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) üzerinden indirip `data/raw/ptbxl/` altına yerleştirin.

---

## Model Eğitimi

```bash
python train_full.py
```
- Eğitim sırasında her epoch sonunda eğitim ve doğrulama için loss, F1, precision, recall metrikleri ekrana yazdırılır.
- Model ağırlıkları `models_artifacts/best_model.pt` dosyasına kaydedilir.

---

## Model Değerlendirme

```bash
python evaluate_model.py
```
- Test sonunda F1, precision, recall, ROC-AUC gibi metrikler ekrana yazdırılır.

---

## Web Uygulaması

Web arayüzü ile sinyal veya görsel yükleyerek analiz yapmak için:
```bash
cd apps/web
# Gerekli bağımlılıkları yükleyin ve başlatın (ör. npm, yarn, vs.)
```

---

## Sıkça Sorulan Sorular

- **Veri seti neden repoda yok?**  
  PTB-XL veri seti lisans gereği repoya eklenmemiştir. Lütfen resmi kaynaktan indiriniz.

- **Modeli tekrar eğitmek zorunda mıyım?**  
  Hayır, eğitilmiş model dosyasını kullanarak doğrudan tahmin yapabilirsiniz.

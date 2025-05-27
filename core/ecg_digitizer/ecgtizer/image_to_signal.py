import cv2
import numpy as np

def decode_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Her türlü (jpg, jpeg, png, bmp, vs.) görseli bytes olarak alıp numpy array'e çevirir.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Görsel decode edilemedi! (Format desteklenmiyor veya bozuk)")
    return img

def extract_ecg_signal_from_image(image: np.ndarray) -> np.ndarray:
    """
    Görselden EKG sinyalini çıkarır.
    Args:
        image (np.ndarray): BGR formatında (cv2.imread ile okunan) görsel.
    Returns:
        np.ndarray: Çıkarılan sinyal (1D array)
    """
    if image is None:
        raise ValueError("Görsel None geldi!")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    signal = []
    for x in range(binary.shape[1]):
        y_vals = np.where(binary[:, x] > 0)[0]
        if len(y_vals) > 0:
            signal.append(y_vals[-1])
        else:
            signal.append(np.nan)
    signal = np.array(signal)
    signal = (np.nanmax(signal) - signal)
    return signal

def save_signal_from_image_bytes(image_bytes: bytes, output_path: str):
    """
    API'den gelen görsel bytes'ını alır, EKG sinyalini çıkarır ve dijitalleşmiş halini output_path'e kaydeder.
    Args:
        image_bytes (bytes): Görselin bytes hali (örn. API'den gelen dosya).
        output_path (str): Dijitalleşmiş sinyalin kaydedileceği dosya yolu (.npy veya .csv önerilir).
    """
    img = decode_image_from_bytes(image_bytes)
    signal = extract_ecg_signal_from_image(img)
    if output_path.endswith('.npy'):
        np.save(output_path, signal)
    elif output_path.endswith('.csv'):
        np.savetxt(output_path, signal, delimiter=',')
    else:
        raise ValueError("output_path .npy veya .csv ile bitmeli!")
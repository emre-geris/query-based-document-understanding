# Bitirme Projesi – Sorguya Göre Metin / Görüntü / Hibrit Yöntem Seçen Doküman Anlama Sistemi

Hasan Emre Geriş · Danışman: Dr. Öğr. Üyesi Hikmet CANLI

## Klasör yapısı

| Klasör | Ne konur | Kural |
|---|---|---|
| `arastirma/` | Literatür Excel'i, plan ve şablon belgeleri | Elle hazırlanan, teslim edilen belgeler |
| `kod/haftaXX_konu/` | O haftanın deney kodu | Kod burada durur, çıktı üretmez buraya |
| `ciktilar/haftaXX_konu/` | Kodun ürettiği her şey (png, txt, csv, json) | Silinse de kod yeniden üretir; elle düzenlenmez |
| `notlar/` | Bulgular (`haftaXX_..._bulgular.md`) ve `deney_gunlugu.md` | Senin yorumların; en değerli kısım |

Hafta numarası plan dokümanındaki haftadır (ör. OCR denemesi planın Hafta 2 maddesi
"OCR tabanlı yaklaşımlar / metin tabanlı modellerin zayıf yönleri" ile ilgilidir).
İleride birden çok haftada kullanılacak ortak kod (ör. ANLS hesabı, veri yükleme) çıkınca
`kod/ortak/` klasörüne taşınır.

## Kurulum (Windows, bir kez)

Sanal ortamı (venv) **OneDrive dışında** tut: yüzlerce MB'lık kütüphaneyi senkronlamaya
çalışır ve taşınan venv bozulur.

```powershell
py -3.13 -m venv C:\Users\emreg\venvs\bitirme
C:\Users\emreg\venvs\bitirme\Scripts\Activate.ps1
cd C:\Users\emreg\OneDrive\Desktop\BitirmeProjesi
pip install -r requirements.txt
```

Tesseract programı ayrıca kurulur: `winget install UB-Mannheim.TesseractOCR`

## Çalıştırma (proje kökünden)

```powershell
python kod/hafta02_ocr_sinirlari/ocr_psm_deneyi.py --n 20
```

## Her deneyden sonra

1. `notlar/deney_gunlugu.md`'ye bir satır ekle (tarih, ne denendi, ayarlar, sonuç).
2. Yorumunu `notlar/haftaXX_..._bulgular.md`'ye yaz; sayıları `ciktilar/.../sonuclar.csv`'den al.

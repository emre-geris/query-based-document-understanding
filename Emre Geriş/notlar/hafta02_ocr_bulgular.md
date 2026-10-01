# Ders 1 — OCR'nin sınırı: bulgular

DocVQA validation, ilk 20 soru, Tesseract 5.4.

## 1. OCR iki ayrı katmanda çöküyor

| Katman | Ne olur | Örnek |
|---|---|---|
| **Tanıma** | Yazıyı görür, yanlış okur | `CALIFORNIA` → `CALIBORNIS`, `0.28` → `8.28` |
| **Segmentasyon** | Yazıya hiç bakmaz | `WHILE YOU WERE OUT` çıktıda yok — okunabilir olduğu hâlde "metin değil" sayılıp elenmiş |

İkincisi daha sinsi: çıktıya bakınca ikisi de "eksik metin" gibi görünür, ama biri okuma hatası, diğeri bakmama hatası.

## 2. PSM (sayfa segmentasyon modu) bakma kararını değiştiriyor

`ornek_2.png`, aynı görsel, aynı motor:

- **PSM 3** (varsayılan) → 5 kelime, sadece antet
- **PSM 6** (düzen analizi yok) → 91 kelime, form etiketlerinin hepsi

Metin hep oradaydı. Varsayılan mod onu elemişti.

## 3. Ama daha çok metin, daha çok cevap demek değil

| Mod | Başarı | Toplam kelime |
|---|---|---|
| PSM 3 | **12/20 (%60)** | 3534 |
| PSM 4 | 10/20 (%50) | 3601 |
| PSM 6 | 9/20 (%45) | **4841** |
| PSM 11 | 9/20 (%45) | 4093 |
| PSM 12 | 9/20 (%45) | 4129 |

En çok metin çıkaran mod en düşük skoru aldı. Örneklerin çoğu `table/list`; PSM 6 sayfayı tek blok sayıp sütunları birbirine karıştırdığı için PSM 3'ün doğru okuduğu hücreleri bozuyor. Kapsam arttı, doğruluk düştü.

## Sonuç

OCR ayarlarını kurcalamak kapsama sorununu çözer, doğruluk sorununa dokunmaz. `görsel → OCR → metin → model` zincirinde bilgi ilk halkada kaybolursa sonraki hiçbir halka onu geri getiremez.

Ölçülen oranlar bir **tavan**: cevabın metinde geçmesi, bir modelin onu bulacağını göstermez.

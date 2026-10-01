# Deney Günlüğü

Her çalıştırma için bir satır. Sayılar `ciktilar/.../sonuclar.csv` ve `calisma_bilgisi.json`'dan alınır.

| Tarih | Deney | Kod | Ayarlar | Sonuç (kısa) | Not |
|---|---|---|---|---|---|
| 2026-09-30 | OCR – PSM karşılaştırması (ilk çalışma) | `kod/hafta02_ocr_sinirlari/ocr_psm_deneyi.py`'nin önceki sürümü | DocVQA val, ilk 20 soru, Tesseract 5.4, PSM 3/4/6/11/12 | PSM 3: 12/20, PSM 4: 10/20, PSM 6/11/12: 9/20 (alt-dizi eşleşmesi) | 20 soru yalnız 7 farklı sayfadan geliyor. Güncel sürüm aynı sayıları `alt_dizi_eslesme` sütununda yeniden üretir; eski dosyalar silindi. |

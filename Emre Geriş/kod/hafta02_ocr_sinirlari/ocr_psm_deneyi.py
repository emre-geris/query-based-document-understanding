"""
HAFTA 2 – OCR'nin sınırı: Tesseract sayfa segmentasyon modlarının (PSM) karşılaştırılması
======================================================================================

Plan bağlantısı: Hafta 2 – "OCR tabanlı yaklaşımların incelenmesi" ve
"metin tabanlı modellerin güçlü/zayıf yönlerinin karşılaştırılması".

Ne yapar?
  1) DocVQA doğrulama kümesinden ilk N soruyu indirir (internet gerekir, GPU gerekmez).
  2) Her BENZERSİZ sayfayı bir kez PNG olarak kaydeder. (DocVQA'da aynı sayfaya birden çok
     soru sorulur; aynı sayfayı tekrar tekrar OCR'lamak hem zaman kaybı hem de sonucu
     'daha çok veri varmış gibi' gösterir.)
  3) Her sayfayı Tesseract'ın FARKLI PSM modlarıyla okur, her birini ayrı .txt kaydeder.
  4) Doğru cevabın hangi modun metninde geçtiğini karşılaştırır.
  5) Tüm sonuçları sonuclar.csv'ye, çalışma bilgilerini calisma_bilgisi.json'a yazar
     -> bulgular elle kopyalanmaz, dosyadan okunur (yeniden üretilebilirlik).

NEDEN PSM ÖNEMLİ?
  Tesseract sayfayı iki aşamada işler. Önce DÜZEN ANALİZİ: sayfayı bloklara böler ve her
  bloğu "metin" / "metin değil" diye etiketler. Sonra yalnızca metin dediği blokları okur.
  Bu ilk karar yanlış olursa, okunabilir yazılar hiç denenmeden elenir.
     3  : tam otomatik düzen analizi   (pytesseract'ın varsayılanı)
     4  : tek sütun, değişken satır boyu
     6  : düzen analizi YOK - tüm sayfayı tek bir metin bloğu say
    11  : dağınık metin - sırayı umursamadan bulabildiğin kadar yazı bul
    12  : dağınık metin + düzen analizi

KURULUM: proje kökündeki README.md'ye bak (pip install -r requirements.txt + Tesseract).

ÇALIŞTIRMA (proje kökünden):
  python kod/hafta02_ocr_sinirlari/ocr_psm_deneyi.py
  python kod/hafta02_ocr_sinirlari/ocr_psm_deneyi.py --n 50
  python kod/hafta02_ocr_sinirlari/ocr_psm_deneyi.py --psm 3,6

ÇIKTILAR: ciktilar/hafta02_ocr_sinirlari/
  sayfa_<docId>.png, sayfa_<docId>_psm<p>.txt, sonuclar.csv, calisma_bilgisi.json
"""

import argparse
import csv
import json
import os
import shutil
import string
import sys
from datetime import datetime
from pathlib import Path

import pytesseract                      # Tesseract OCR motorunu Python'dan çağıran köprü
from datasets import load_dataset       # Hugging Face veri setlerini indirir

# Windows konsolu cp1252 olabilir -> Türkçe karakterler UnicodeEncodeError verir.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

VARSAYILAN_PSM = [3, 4, 6, 11, 12]
PROJE_KOKU = Path(__file__).resolve().parents[2]          # kod/hafta02_.../bu_dosya -> proje kökü
VARSAYILAN_CIKTI = PROJE_KOKU / "ciktilar" / "hafta02_ocr_sinirlari"


# ---------------------------------------------------------------------------
# 0) Tesseract'ı bul (Windows'ta PATH'e eklenmemiş olabilir)
# ---------------------------------------------------------------------------
def tesseract_hazirla():
    if shutil.which("tesseract"):
        return
    windows_yolu = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(windows_yolu):
        pytesseract.pytesseract.tesseract_cmd = windows_yolu
        return
    raise SystemExit(
        "Tesseract bulunamadı. README.md'deki kurulum adımlarını uygula.\n"
        "Windows'ta farklı klasöre kurduysan 'windows_yolu' değişkenini düzenle."
    )


# ---------------------------------------------------------------------------
# 1) Cevap metinde geçiyor mu?
#    İki yöntem tutuyoruz ve farkı raporluyoruz:
#     - alt_dizi : eski yöntem. "1" gibi kısa cevaplar "2019" içinde de 'bulunur' -> yanlış EVET.
#     - kelime   : yeni yöntem. Cevabın kelimeleri OCR metninde ardışık TAM KELİMELER olarak
#                  geçmeli. Kısa cevaplarda sahte eşleşmeyi önler.
# ---------------------------------------------------------------------------
def normalize(metin: str) -> str:
    metin = metin.lower()
    metin = metin.translate(str.maketrans("", "", string.punctuation))
    return " ".join(metin.split())


def alt_dizi_eslesme(cevaplar, ocr_metni) -> bool:
    ocr = normalize(ocr_metni)
    return any(normalize(c) and normalize(c) in ocr for c in cevaplar)


def kelime_eslesme(cevaplar, ocr_metni) -> bool:
    ocr = normalize(ocr_metni).split()
    for c in cevaplar:
        hedef = normalize(c).split()
        if not hedef:
            continue
        n = len(hedef)
        if any(ocr[i:i + n] == hedef for i in range(len(ocr) - n + 1)):
            return True
    return False


# ---------------------------------------------------------------------------
# 2) Ana akış
# ---------------------------------------------------------------------------
def main(n: int, psm_listesi, cikti: Path):
    tesseract_hazirla()
    cikti.mkdir(parents=True, exist_ok=True)

    print("DocVQA indiriliyor (streaming – yalnızca gereken kadar iner)...")
    print("Denenecek PSM modları:", ", ".join(str(p) for p in psm_listesi))
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation", streaming=True)

    ocr_onbellek = {}      # (docId, psm) -> OCR metni   : aynı sayfa ikinci kez OCR'lanmaz
    satirlar = []          # sonuclar.csv'ye gidecek satırlar

    for i, ornek in enumerate(ds, start=1):
        if i > n:
            break

        doc_id = ornek["docId"]
        goruntu = ornek["image"]
        png = cikti / f"sayfa_{doc_id}.png"
        yeni_sayfa = (doc_id, psm_listesi[0]) not in ocr_onbellek
        if not png.exists():
            goruntu.save(png)

        print("\n" + "=" * 78)
        print(f"SORU {i}  (sayfa {doc_id}{'' if yeni_sayfa else ' – daha önce görülen sayfa'})")
        print("SORU        :", ornek["question"])
        print("DOĞRU CEVAP :", ornek["answers"])
        print("SORU TİPİ   :", ", ".join(ornek["question_types"]))
        print()
        print(f"  {'PSM':<6}{'Kelime':<9}{'Kelime eşl.':<14}{'Alt dizi eşl.':<15}Dosya")
        print("  " + "-" * 64)

        for p in psm_listesi:
            anahtar = (doc_id, p)
            txt = cikti / f"sayfa_{doc_id}_psm{p}.txt"
            if anahtar not in ocr_onbellek:
                ocr_onbellek[anahtar] = pytesseract.image_to_string(goruntu, config=f"--psm {p}")
                txt.write_text(ocr_onbellek[anahtar], encoding="utf-8")
            ocr_metni = ocr_onbellek[anahtar]

            kelime_sayisi = len(ocr_metni.split())
            k_esl = kelime_eslesme(ornek["answers"], ocr_metni)
            a_esl = alt_dizi_eslesme(ornek["answers"], ocr_metni)
            print(f"  {p:<6}{kelime_sayisi:<9}{('EVET' if k_esl else 'HAYIR'):<14}"
                  f"{('EVET' if a_esl else 'HAYIR'):<15}{txt.name}")

            satirlar.append({
                "soru_no": i, "question_id": ornek["questionId"], "doc_id": doc_id,
                "soru": ornek["question"], "cevaplar": " | ".join(ornek["answers"]),
                "soru_tipi": ", ".join(ornek["question_types"]), "psm": p,
                "ocr_kelime": kelime_sayisi, "kelime_eslesme": int(k_esl),
                "alt_dizi_eslesme": int(a_esl),
            })

    if not satirlar:
        raise SystemExit("Hiç örnek işlenmedi.")

    # --- Dosyaya yaz ---------------------------------------------------------
    csv_yolu = cikti / "sonuclar.csv"
    with csv_yolu.open("w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: Excel Türkçe'yi doğru açar
        w = csv.DictWriter(f, fieldnames=list(satirlar[0].keys()))
        w.writeheader()
        w.writerows(satirlar)

    soru_sayisi = len({s["soru_no"] for s in satirlar})
    sayfa_sayisi = len({s["doc_id"] for s in satirlar})
    bilgi = {
        "tarih": datetime.now().isoformat(timespec="seconds"),
        "veri_seti": "lmms-lab/DocVQA (DocVQA, validation, streaming, ilk N soru)",
        "soru_sayisi": soru_sayisi,
        "benzersiz_sayfa_sayisi": sayfa_sayisi,
        "psm_modlari": psm_listesi,
        "tesseract_surumu": str(pytesseract.get_tesseract_version()),
        "python_surumu": sys.version.split()[0],
    }
    (cikti / "calisma_bilgisi.json").write_text(json.dumps(bilgi, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Özet ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"ÖZET — {soru_sayisi} soru, yalnızca {sayfa_sayisi} farklı sayfa")
    if sayfa_sayisi < soru_sayisi:
        print("  DİKKAT: Aynı sayfadaki sorular birbirinden bağımsız değildir. Sonuçlar sayfa\n"
              "  sayısı kadar 'bağımsız kanıt' içerir; kesin yargı için --n'yi artırın.")
    print(f"\n  {'PSM':<6}{'Kelime eşl.':<16}{'Alt dizi eşl.':<16}{'Sahte EVET':<12}Toplam kelime*")
    for p in psm_listesi:
        s = [r for r in satirlar if r["psm"] == p]
        k = sum(r["kelime_eslesme"] for r in s)
        a = sum(r["alt_dizi_eslesme"] for r in s)
        sahte = sum(1 for r in s if r["alt_dizi_eslesme"] and not r["kelime_eslesme"])
        kelime = sum(len(ocr_onbellek[(d, p)].split()) for d in {r["doc_id"] for r in s})
        print(f"  {p:<6}{f'{k}/{len(s)} (%{k/len(s)*100:.0f})':<16}{f'{a}/{len(s)}':<16}{sahte:<12}{kelime}")
    print("  * Toplam kelime benzersiz sayfalar üzerinden (aynı sayfa bir kez sayılır).")
    print("  'Sahte EVET' = eski alt-dizi yönteminin EVET dediği ama cevabın tam kelime olarak geçmediği durum.")

    print(f"\nDosyalar: {cikti}")
    print("  sonuclar.csv -> Excel'de açıp soru tipine göre filtreleyebilirsin.")
    print("\nNot: 'EVET' cevabın metinde GEÇTİĞİNİ gösterir, bir modelin onu BULACAĞINI değil.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20, help="Denenecek soru sayısı")
    p.add_argument("--psm", default=",".join(map(str, VARSAYILAN_PSM)),
                   help="Denenecek PSM modları, virgülle ayrık. Örn: --psm 3,6")
    p.add_argument("--cikti", default=str(VARSAYILAN_CIKTI), help="Çıktı klasörü")
    a = p.parse_args()
    main(a.n, [int(x) for x in a.psm.split(",") if x.strip()], Path(a.cikti))

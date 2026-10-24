# NAIM ile Eksik Klinik Veriler Üzerinde Sınıflandırma (UCI CKD)

Bu depo, **NAIM (Not Another Imputation Method)** yöntemini yazarların yayımladığı kodla
**UCI Chronic Kidney Disease** verisine uygular ve NAIM'i medyan / en sık değerle doldurma yapan
**SVM, KNN ve Random Forest** ile farklı eksiklik oranlarında karşılaştırır.

- Makale: C. M. Caruso, P. Soda, V. Guarrasi, *Not another imputation method: A transformer-based model
  for missing values in tabular datasets*, AI Open 7 (2026) 96–122. doi:10.1016/j.aiopen.2026.02.005
- NAIM kodu: [github.com/cosbidev/NAIM](https://github.com/cosbidev/NAIM)
- Veri: [UCI Chronic Kidney Disease](https://archive.ics.uci.edu/dataset/336) (doi:10.24432/C5G020)

## Deney özeti

| Adım | Ayrıntı |
| --- | --- |
| Veri | 400 hasta, 24 özellik (11 sayısal, 13 kategorik), 250 ckd / 150 notckd, hücrelerin %10,5'i eksik |
| Ayrım | Stratified 280 / 60 / 60 (train / validation / test), seed 42 ile bir kez yapılır ve sabit kalır |
| Ek eksiklik | Her sette gözlenen değerlerin ek olarak %0, %10, %30, %50'si rastgele gizlenir; bütün modeller aynı maskeyi kullanır |
| Tekrar | Her oran 42, 123 ve 2026 seed'leriyle tekrarlanır (ortalama ± standart sapma) |
| Modeller | NAIM (eksikler korunur) · SVM, KNN, RF (sayısal eksik → train medyanı, kategorik eksik → train'de en sık değer) |
| Metrikler | Accuracy, macro-F1, sınıf bazlı recall (ek olarak AUC) |

## Kurulum

Python 3.10 veya üzeri gerekir. GPU gerekmez; her şey CPU'da çalışır.

```bash
git clone https://github.com/MuhammedBurakKizilok/naim-ckd
cd naim-ckd

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Not: Komutlar sanal ortam açıkken `python3` ile çalıştırılır. `pip install` başarılı olduğu hâlde
> `ModuleNotFoundError: No module named 'numpy'` görürseniz, kabuğunuzdaki `python`/`python3` başka bir
> kuruluma yönleniyordur (alias, conda, pyenv). Bu durumda komutları `.venv/bin/python src/...` şeklinde çalıştırın.

Veri dosyası `data/chronic_kidney_disease_full.arff` depoda bulunur (UCI, CC BY 4.0).
Yazarların NAIM kodu `NAIM/` klasöründedir (MIT lisansı, commit `NAIM/COMMIT.txt` içinde).

## Hızlı deneme (~1 dakika)

Tam deneyden önce kurulumu doğrulamak için tek bir oran ve tek bir seed ile çalıştırın.
Sonuçlar ayrı bir klasöre yazılır, depodaki sonuçlar değişmez:

```bash
python3 src/run_experiments.py --ratios 0.3 --seeds 42 --out results_quicktest
```

Beklenen çıktı (değerler `results/runs.csv` ile aynıdır):

```
[ratio 0.30 seed 42] NAIM  acc=0.983 f1=0.982 best_epoch=79
[ratio 0.30 seed 42] SVM   acc=0.967 f1=0.965
[ratio 0.30 seed 42] KNN   acc=0.983 f1=0.982
[ratio 0.30 seed 42] RF    acc=0.983 f1=0.982
```

scikit-learn 1.9 ve sonrası `SVC(probability=True)` için bir `FutureWarning` yazdırır; sonuçları etkilemez.

## Deneylerin tamamını çalıştırma

Komutlar proje klasöründen sırayla çalıştırılır.

```bash
# 1) Veri analizi: değişkenler, eksiklik oranları, sınıf dağılımı, sabit ayrım 
python3 src/analyze_data.py

# 2) Ana deney: 4 oran × 3 seed × 4 model, NAIM yazarların kodundaki gibi val AUC ile 
python3 src/run_experiments.py

# 3) Ek deney: NAIM'in erken durdurma ölçütü validation loss 
python3 src/run_experiments.py --stop-metric loss --naim-only --tag _valloss

# 4) Tablolar ve grafikler (ortalama ± standart sapma)
python3 src/summarize_results.py

# 5) İsteğe bağlı: eksikliğin sınıf bilgisi taşıyıp taşımadığı ve NAIM'deki eksik token'lar 
python3 src/missingness_analysis.py
```

Çalışma sırasında her NAIM eğitimi için ekrana bir satır yazılır.

## Çıktılar

| Dosya | İçerik |
| --- | --- |
| `results/data_analysis/data_analysis.md` | Değişken tablosu, eksiklik oranları, ayrım özeti |
| `results/data_analysis/*.png` | Sınıfa göre eksiklik ve sınıf dağılımı grafikleri |
| `results/runs.csv` | Her (oran, seed, model) için test metrikleri; NAIM için en iyi epoch |
| `results/runs_valloss.csv` | Validation loss ile eğitilen NAIM'in sonuçları |
| `results/results_table.md` | Ortalama ± standart sapma sonuç tablosu |
| `results/performance_vs_missing.png` | Eksiklik oranına göre accuracy, macro-F1 ve recall grafikleri |
| `results/test_predictions*.csv` | Test hastası bazında tahmin olasılıkları |
| `results/naim_training_history*.csv` | NAIM'in epoch bazında eğitim kaybı, val AUC, val loss |
| `results/missingness_analysis/` | Ek analiz sonuçları |

## Proje yapısı

```
naim-ckd/
├── NAIM/                     yazarların kodu (değiştirilmedi)
├── data/                     chronic_kidney_disease_full.arff
├── src/
│   ├── data_utils.py         veri okuma, sabit ayrım, ön işleme
│   ├── analyze_data.py       veri analizi
│   ├── run_experiments.py    ek eksiklik üretimi, NAIM ve baseline eğitimi, metrikler
│   ├── summarize_results.py  ortalama ± std tabloları ve grafikler
│   └── missingness_analysis.py
├── results/                  deneylerin çıktıları
└── requirements.txt
```

## Kaynak kodda yapılan değişiklikler

- NAIM modeli (`naim.py`, `tabular_tokenizer.py`), regularization (`missing_augmentation`) ve L1/L2
  kayıpları **değiştirilmeden**, `NAIM/` klasöründeki dosyalardan doğrudan yüklenir.
- Repodaki Hydra tabanlı pipeline 5 katlı çapraz doğrulama yapar, validation setine eksiklik eklemez ve
  CKD yapılandırması içermez. Bu yüzden deney altyapısı (veri okuma, sabit 70/15/15 ayrım, ortak maskeler,
  seed tekrarları, metrikler) `src/` altında yeniden yazıldı.
- `train_naim` fonksiyonu yazarların `TrainManager` mantığının sadeleştirilmiş bir kopyasıdır
  (Adam, batch 32, LR 1e-3 → 1e-7, 50 epoch ısınma, 50 epoch sabırlı erken durdurma, L1 ve L2 cezası 1e-4).
  Varsayılan ölçüt yazarların kodundaki gibi validation AUC'dir; `--stop-metric loss` validation loss kullanır.
- Ön işleme istatistikleri (min-max, medyan, en sık değer) yalnızca train setinden öğrenilir.

## Tekrarlanabilirlik notu

Ayrım, gizleme maskeleri ve seed'ler sabittir. PyTorch'un CPU hesapları işletim sistemi ve sürüme göre
çok küçük sayısal farklar üretebildiği için NAIM sonuçları farklı bir makinede birkaç hastalık fark
gösterebilir; SVM, KNN ve RF sonuçları aynı çıkmalıdır.

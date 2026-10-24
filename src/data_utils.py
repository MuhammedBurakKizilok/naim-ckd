"""
UCI Chronic Kidney Disease (CKD) veri setini okuma, şema ve ön işleme yardımcıları.

Ham ARFF dosyasında bilinen yazım sorunları var (fazladan virgül, sekme karakterleri,
'\tno' gibi değerler). Bu dosya onları temizler ve eksik değerleri ('?') NaN olarak tutar.
"""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ARFF başlığındaki resmi tanım: 11 sayısal, 13 kategorik (nominal) özellik + sınıf.
NUMERICAL = ["age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]
CATEGORICAL = {
    "sg":    ["1.005", "1.010", "1.015", "1.020", "1.025"],
    "al":    ["0", "1", "2", "3", "4", "5"],
    "su":    ["0", "1", "2", "3", "4", "5"],
    "rbc":   ["normal", "abnormal"],
    "pc":    ["normal", "abnormal"],
    "pcc":   ["present", "notpresent"],
    "ba":    ["present", "notpresent"],
    "htn":   ["yes", "no"],
    "dm":    ["yes", "no"],
    "cad":   ["yes", "no"],
    "appet": ["good", "poor"],
    "pe":    ["yes", "no"],
    "ane":   ["yes", "no"],
}
# Dosyadaki özellik sırası
FEATURES = ["age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc",
            "sod", "pot", "hemo", "pcv", "wc", "rc", "htn", "dm", "cad", "appet", "pe", "ane"]
TARGET = "class"
CLASSES = ["notckd", "ckd"]          # 0 = notckd, 1 = ckd

DESCRIPTIONS = {
    "age": "Yaş (yıl)", "bp": "Kan basıncı (mm/Hg)", "sg": "İdrar özgül ağırlığı",
    "al": "Albümin", "su": "Şeker (idrar)", "rbc": "İdrarda kırmızı kan hücresi",
    "pc": "Püs hücresi", "pcc": "Püs hücresi kümeleri", "ba": "Bakteri",
    "bgr": "Rastgele kan şekeri (mg/dl)", "bu": "Kan üre (mg/dl)", "sc": "Serum kreatinin (mg/dl)",
    "sod": "Sodyum (mEq/L)", "pot": "Potasyum (mEq/L)", "hemo": "Hemoglobin (g)",
    "pcv": "Paketlenmiş hücre hacmi", "wc": "Beyaz kan hücresi sayısı (hücre/mm³)",
    "rc": "Kırmızı kan hücresi sayısı (milyon/mm³)", "htn": "Hipertansiyon", "dm": "Diyabet",
    "cad": "Koroner arter hastalığı", "appet": "İştah", "pe": "Ayakta ödem", "ane": "Anemi",
}


def _canon(value: str, col: str) -> str:
    """Kategorik değeri normalize eder: '1.01' -> '1.010' gibi."""
    if col == "sg":
        return f"{float(value):.3f}"
    if col in ("al", "su"):
        return str(int(float(value)))
    return value


def load_ckd(path: str) -> pd.DataFrame:
    """ARFF dosyasını okur ve temiz bir DataFrame döner (eksik değerler NaN)."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().lower() == "@data") + 1

    rows, problems = [], []
    for ln, line in enumerate(lines[start:], start=start + 1):
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        tokens = [re.sub(r"\s+", "", t) for t in line.split(",")]
        if len(tokens) > 25:                       # fazladan boş alanlar (ör. ',,')
            tokens = [t for t in tokens if t != ""]
        if len(tokens) != 25:
            problems.append((ln, len(tokens)))
            continue
        rows.append(tokens)
    if problems:
        raise ValueError(f"Okunamayan satırlar: {problems}")

    df = pd.DataFrame(rows, columns=FEATURES + [TARGET]).replace({"?": np.nan, "": np.nan})
    for c in NUMERICAL:
        df[c] = pd.to_numeric(df[c], errors="raise")
    for c, cats in CATEGORICAL.items():
        df[c] = df[c].map(lambda v: _canon(v, c) if isinstance(v, str) else v)
        bad = set(df[c].dropna()) - set(cats)
        if bad:
            raise ValueError(f"{c} için beklenmeyen değerler: {bad}")
    bad = set(df[TARGET]) - set(CLASSES)
    if bad:
        raise ValueError(f"Beklenmeyen sınıf değerleri: {bad}")
    df.index.name = "patient_id"
    return df


# ----------------------------------------------------------------------------------
# Ön işleme. Tüm istatistikler SADECE eğitim setinden öğrenilir (veri sızıntısı yok).
# ----------------------------------------------------------------------------------
class NaimPreprocessor:
    """NAIM için: sayısalları [0,1]'e ölçekler, kategorikleri indekse çevirir, NaN'ları KORUR."""

    def fit(self, X: pd.DataFrame):
        self.min_ = X[NUMERICAL].min()
        self.max_ = X[NUMERICAL].max()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        out = X[FEATURES].copy()
        rng = (self.max_ - self.min_).replace(0, 1.0)
        out[NUMERICAL] = (X[NUMERICAL] - self.min_) / rng
        for c, cats in CATEGORICAL.items():
            out[c] = X[c].map({v: i for i, v in enumerate(cats)})   # NaN -> NaN kalır
        return out.astype(float).values

    @property
    def cat_idxs(self):
        return [FEATURES.index(c) for c in CATEGORICAL]

    @property
    def cat_dims(self):
        return [len(CATEGORICAL[FEATURES[i]]) for i in self.cat_idxs]


class BaselinePreprocessor:
    """SVM/KNN/RF için: medyan / en sık değer ile doldurma, [0,1] ölçekleme, one-hot."""

    def fit(self, X: pd.DataFrame):
        self.median_ = X[NUMERICAL].median()
        self.mode_ = {c: X[c].mode(dropna=True).iloc[0] for c in CATEGORICAL}
        filled = X[NUMERICAL].fillna(self.median_)
        self.min_, self.max_ = filled.min(), filled.max()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        num = X[NUMERICAL].fillna(self.median_)
        num = (num - self.min_) / (self.max_ - self.min_).replace(0, 1.0)
        parts = [num.values]
        for c, cats in CATEGORICAL.items():
            col = X[c].fillna(self.mode_[c])
            parts.append(np.stack([(col == v).astype(float).values for v in cats], axis=1))
        return np.concatenate(parts, axis=1)


def encode_labels(y: pd.Series) -> np.ndarray:
    return y.map({c: i for i, c in enumerate(CLASSES)}).astype(int).values


SPLIT_SEED = 42   # 70/15/15 ayrımı bu seed ile BİR KEZ yapılır ve bütün deneylerde sabit kalır


def fixed_split(df):
    """Stratified ~70/15/15; seed sabit, tüm deneylerde aynı hastalar."""
    y = df["class"]
    tr, tmp = train_test_split(df.index, test_size=0.30, stratify=y, random_state=SPLIT_SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y.loc[tmp], random_state=SPLIT_SEED)
    return {"train": sorted(tr), "val": sorted(va), "test": sorted(te)}

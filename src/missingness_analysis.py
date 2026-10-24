"""
Ek analiz: Eksiklik bu veride bilgi taşıyor mu ve NAIM bunu görebiliyor mu?

1) Sadece "hangi özellik eksik" bilgisiyle (değerler olmadan) lojistik regresyon -> test başarımı.
2) Eğitilmiş bir NAIM'de (ek gizleme yok, seed 42) son encoder katmanından sonra eksik token'lar
   farklı hastalarda ve farklı özelliklerde aynı (sabit) vektör mü? Sınıflandırıcı katmanı tüm
   pozisyonları birleştirdiği için bu sabit vektör "bu özellik eksik" bilgisini taşır.

Kullanım:
    python src/missingness_analysis.py
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
import run_experiments as R  # noqa: E402

ROOT = R.ROOT


def main():
    out_dir = os.path.join(ROOT, "results", "missingness_analysis")
    os.makedirs(out_dir, exist_ok=True)
    df = R.load_ckd(os.path.join(ROOT, "data", "chronic_kidney_disease_full.arff"))
    split = R.fixed_split(df)
    res = {}

    # 1) Sadece eksiklik göstergeleri ile sınıflandırma (orijinal veri)
    M = df[R.FEATURES].isna().astype(float)
    y = R.encode_labels(df["class"])
    yi = {k: y[df.index.get_indexer(v)] for k, v in split.items()}
    lr = LogisticRegression(max_iter=1000).fit(M.loc[split["train"]], yi["train"])
    p = lr.predict(M.loc[split["test"]])
    res["sadece_eksiklik_ile_LR"] = dict(test_accuracy=accuracy_score(yi["test"], p),
                                        test_macro_f1=f1_score(yi["test"], p, average="macro"),
                                        cogunluk_sinifi_accuracy=float(max(yi["test"].mean(), 1 - yi["test"].mean())))

    # 2) Eğitilmiş NAIM'de eksik token'lar
    cond = R.make_condition(df, split, 0.0, 42)
    model, prep, info, _ = R.train_naim(cond, 42)
    Xte = prep.transform(cond["test"][0])
    captured = {}
    h = model.encoder[-1].register_forward_hook(lambda m, i, o: captured.__setitem__("last", o.detach()))
    model.eval()
    with torch.no_grad():
        model(torch.tensor(Xte, dtype=torch.float32))
    h.remove()
    tok = captured["last"].numpy()                    # (hasta, özellik, d_token)
    miss = np.isnan(Xte)
    mt, pt = tok[miss], tok[~miss]
    res["egitilmis_NAIM_eksik_token"] = dict(
        eksik_token_sayisi=int(miss.sum()),
        eksik_tokenlar_arasi_max_fark=float(np.abs(mt - mt[0]).max()) if len(mt) else None,
        eksik_token_vektoru=mt[0].round(4).tolist() if len(mt) else None,
        eksik_token_normu=float(np.linalg.norm(mt[0])) if len(mt) else None,
        mevcut_token_ortalama_normu=float(np.linalg.norm(pt, axis=1).mean()),
        best_epoch=info["best_epoch"])

    with open(os.path.join(out_dir, "missingness_analysis.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""
Veri analizi: değişkenler, türler, eksiklik oranları, sınıf dağılımı ve sabit 70/15/15 ayrımı.
Çıktılar: results/data_analysis/ altında tablolar (CSV/Markdown) ve grafikler (PNG).

Kullanım:
    python src/analyze_data.py --data data/chronic_kidney_disease_full.arff
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import CATEGORICAL, DESCRIPTIONS, FEATURES, NUMERICAL, fixed_split, load_ckd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C_CKD, C_NOT = "#c0392b", "#2e86c1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "chronic_kidney_disease_full.arff"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "data_analysis"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_ckd(args.data)
    X = df[FEATURES]
    ckd, notckd = df["class"] == "ckd", df["class"] == "notckd"

    # 1) Değişken tablosu
    rows = []
    for c in FEATURES:
        is_num = c in NUMERICAL
        obs = X[c].dropna()
        rows.append(dict(
            ozellik=c, aciklama=DESCRIPTIONS[c], tur="sayısal" if is_num else "kategorik",
            degerler=(f"{obs.min():g} – {obs.max():g} (medyan {obs.median():g})" if is_num
                      else " / ".join(CATEGORICAL[c])),
            eksik_sayi=int(X[c].isna().sum()), eksik_oran=round(X[c].isna().mean() * 100, 1),
            eksik_ckd=round(X.loc[ckd, c].isna().mean() * 100, 1),
            eksik_notckd=round(X.loc[notckd, c].isna().mean() * 100, 1)))
    var = pd.DataFrame(rows).sort_values("eksik_oran", ascending=False)
    var.to_csv(os.path.join(args.out, "variables.csv"), index=False)

    # 2) Özet sayılar
    per_row = X.isna().sum(axis=1)
    split = fixed_split(df)
    split_rows = []
    for name, idx in split.items():
        s = df.loc[idx]
        split_rows.append(dict(set=name, n=len(s), ckd=int((s["class"] == "ckd").sum()),
                               notckd=int((s["class"] == "notckd").sum()),
                               ckd_oran=round((s["class"] == "ckd").mean() * 100, 1),
                               eksik_hucre_orani=round(s[FEATURES].isna().values.mean() * 100, 1)))
    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(os.path.join(args.out, "split_summary.csv"), index=False)

    summary = {
        "hasta_sayisi": len(df), "ozellik_sayisi": len(FEATURES),
        "sayisal": len(NUMERICAL), "kategorik": len(CATEGORICAL),
        "ckd": int(ckd.sum()), "notckd": int(notckd.sum()),
        "toplam_eksik_hucre_orani_%": round(X.isna().values.mean() * 100, 2),
        "eksik_orani_ckd_%": round(X[ckd].isna().values.mean() * 100, 2),
        "eksik_orani_notckd_%": round(X[notckd].isna().values.mean() * 100, 2),
        "hic_eksigi_olmayan_hasta": int((per_row == 0).sum()),
        "hic_eksigi_olmayan_hasta_ckd": int(((per_row == 0) & ckd).sum()),
        "hic_eksigi_olmayan_hasta_notckd": int(((per_row == 0) & notckd).sum()),
        "hasta_basi_eksik_medyan": float(per_row.median()),
        "hasta_basi_eksik_max": int(per_row.max()),
    }
    pd.Series(summary).to_csv(os.path.join(args.out, "summary.csv"), header=["deger"])

    with open(os.path.join(args.out, "data_analysis.md"), "w") as f:
        f.write("# Veri analizi – UCI Chronic Kidney Disease\n\n")
        for k, v in summary.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n## Değişkenler (eksiklik oranına göre sıralı, %)\n\n")
        f.write(var.to_markdown(index=False))
        f.write("\n\n## Sabit ayrım (stratified, seed=42)\n\n")
        f.write(split_df.to_markdown(index=False))
        f.write("\n")

    # 3) Grafikler
    fig, ax = plt.subplots(figsize=(10, 6))
    order = var["ozellik"].tolist()[::-1]
    y = np.arange(len(order))
    ax.barh(y + 0.2, [var.set_index("ozellik").loc[o, "eksik_ckd"] for o in order], 0.4, color=C_CKD, label="ckd (n=250)")
    ax.barh(y - 0.2, [var.set_index("ozellik").loc[o, "eksik_notckd"] for o in order], 0.4, color=C_NOT, label="notckd (n=150)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{o} ({'S' if o in NUMERICAL else 'K'})" for o in order])
    ax.set_xlabel("Eksik değer oranı (%)")
    ax.set_title("Özellik bazında eksiklik oranı, sınıfa göre  (S: sayısal, K: kategorik)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "missingness_by_feature.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = df["class"].value_counts()[["ckd", "notckd"]]
    axes[0].bar(counts.index, counts.values, color=[C_CKD, C_NOT])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 3, f"{v} (%{v / len(df) * 100:.1f})", ha="center")
    axes[0].set_title("Sınıf dağılımı")
    axes[0].set_ylabel("Hasta sayısı")
    bins = np.arange(0, per_row.max() + 2) - 0.5
    axes[1].hist([per_row[ckd], per_row[notckd]], bins=bins, color=[C_CKD, C_NOT], label=["ckd", "notckd"], stacked=True)
    axes[1].set_title("Hasta başına eksik özellik sayısı")
    axes[1].set_xlabel("Eksik özellik sayısı (24 üzerinden)")
    axes[1].set_ylabel("Hasta sayısı")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "class_and_missing_per_patient.png"), dpi=150)
    plt.close(fig)

    print(open(os.path.join(args.out, "data_analysis.md")).read())


if __name__ == "__main__":
    main()

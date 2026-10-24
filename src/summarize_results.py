"""
results/runs.csv -> ortalama ± standart sapma tabloları ve performans grafikleri.

Kullanım:
    python src/summarize_results.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["NAIM", "NAIM (val-loss)", "SVM", "KNN", "RF"]
COLORS = {"NAIM": "#c0392b", "NAIM (val-loss)": "#e67e22", "SVM": "#2e86c1", "KNN": "#27ae60", "RF": "#8e44ad"}
METRICS = {"accuracy": "Accuracy", "macro_f1": "Macro-F1",
           "recall_ckd": "Recall (ckd)", "recall_notckd": "Recall (notckd)", "auc": "AUC (ek)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    runs = pd.read_csv(os.path.join(args.results, "runs.csv"))
    extra = os.path.join(args.results, "runs_valloss.csv")          # ek deney (varsa)
    if os.path.exists(extra):
        runs = pd.concat([runs, pd.read_csv(extra)], ignore_index=True)
    models = [m for m in MODELS if m in set(runs.model)]

    g = runs.groupby(["ratio", "model"])
    mean, std, n = g[list(METRICS)].mean(), g[list(METRICS)].std(ddof=1), g.size()
    long = []
    for (ratio, model), row in mean.iterrows():
        d = dict(ek_gizleme=f"%{int(ratio * 100)}", model=model, tekrar=int(n.loc[(ratio, model)]))
        for m in METRICS:
            d[m + "_mean"] = row[m]
            d[m + "_std"] = std.loc[(ratio, model), m]
        long.append(d)
    long = pd.DataFrame(long)
    long["model"] = pd.Categorical(long["model"], models, ordered=True)
    long = long.sort_values(["ek_gizleme", "model"], key=lambda s: s.str.strip("%").astype(int) if s.name == "ek_gizleme" else s)
    long.to_csv(os.path.join(args.results, "summary_mean_std.csv"), index=False)

    # Okunabilir tablo: "ort ± std"
    pretty = long[["ek_gizleme", "model"]].copy()
    for m, label in METRICS.items():
        pretty[label] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(long[m + "_mean"], long[m + "_std"])]
    pretty.to_csv(os.path.join(args.results, "results_table.csv"), index=False)
    with open(os.path.join(args.results, "results_table.md"), "w") as f:
        f.write("Test seti (60 hasta) sonuçları – 3 tekrarın (seed 42, 123, 2026) ortalaması ± standart sapması\n\n")
        f.write(pretty.to_markdown(index=False))
        f.write("\n")

    # NAIM eğitim bilgisi
    naim = runs[runs.model.str.startswith("NAIM")].reindex(columns=[
        "model", "ratio", "seed", "best_epoch", "last_epoch", "best_val_auc", "best_val_loss", "seconds",
        "missing_train", "missing_val", "missing_test"])
    naim.to_csv(os.path.join(args.results, "naim_training_info.csv"), index=False)

    # Grafik 1: metrikler vs ek gizleme oranı
    ratios = sorted(runs.ratio.unique())
    main_models = [m for m in models if "val-loss" not in m]      # ana sonuçlar; ek deney tabloda
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
    for ax, m in zip(axes.flat, ["accuracy", "macro_f1", "recall_ckd", "recall_notckd"]):
        for model in main_models:
            s = long[long.model == model].set_index("ek_gizleme")
            xs = [r * 100 for r in ratios]
            ys = [s.loc[f"%{int(r * 100)}", m + "_mean"] for r in ratios]
            es = [s.loc[f"%{int(r * 100)}", m + "_std"] for r in ratios]
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, lw=2.2 if model.startswith("NAIM") else 1.4, ls="--" if "val-loss" in model else "-",
                        color=COLORS[model], label=model)
        ax.set_title(METRICS[m])
        ax.set_xlabel("Ek gizlenen gözlem oranı (%)")
        ax.set_xticks([r * 100 for r in ratios])
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("Test skoru (ort. ± std, 3 seed)")
    axes[0, 0].legend(loc="lower left")
    fig.suptitle("UCI CKD – ek eksiklik oranına göre test performansı", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(args.results, "performance_vs_missing.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Grafik 2: macro-F1 bar chart
    fig, ax = plt.subplots(figsize=(10, 4.4))
    w = 0.8 / len(models)
    for i, model in enumerate(models):
        s = long[long.model == model].set_index("ek_gizleme")
        xs = [j + (i - (len(models) - 1) / 2) * w for j in range(len(ratios))]
        ax.bar(xs, [s.loc[f"%{int(r * 100)}", "macro_f1_mean"] for r in ratios], w,
               yerr=[s.loc[f"%{int(r * 100)}", "macro_f1_std"] for r in ratios], capsize=3,
               color=COLORS[model], label=model)
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels([f"+%{int(r * 100)}" for r in ratios])
    ax.set_xlabel("Ek gizleme oranı")
    ax.set_ylabel("Macro-F1")
    lo = long["macro_f1_mean"].min() - long["macro_f1_std"].max() - 0.03
    ax.set_ylim(max(0, lo), 1.005)
    ax.legend(ncol=len(models), loc="lower left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Macro-F1 (test, ort. ± std)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.results, "macro_f1_bars.png"), dpi=150)
    plt.close(fig)

    print(open(os.path.join(args.results, "results_table.md")).read())
    print(naim.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

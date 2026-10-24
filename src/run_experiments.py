"""
NAIM ile UCI CKD üzerinde eksik veri deneyleri.

- Yazarların NAIM modeli (naim.py, tabular_tokenizer.py), regularization fonksiyonu
  (missing_augmentation) ve L1/L2 kayıpları DEĞİŞTİRİLMEDEN doğrudan repodan yüklenir.
- Eğitim döngüsü, yazarların TrainManager/EarlyStopper/CustomScheduler mantığının
  birebir yeniden yazımıdır (repodaki Hydra pipeline'ı CKD + task kurgusuna uymadığı için).

Kullanım:
    python src/run_experiments.py --data data/chronic_kidney_disease_full.arff
"""
import argparse
import copy
import importlib.util
import json
import os
import sys
import time
import types

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import (CLASSES, FEATURES, BaselinePreprocessor, NaimPreprocessor,  # noqa: E402
                        encode_labels, fixed_split, load_ckd)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAIM_REPO = os.path.join(ROOT, "NAIM")

SEEDS = [42, 123, 2026]
MISSING_RATIOS = [0.0, 0.10, 0.30, 0.50]   # 0.0 = ek gizleme yok (referans)


# ----------------------------------------------------------------------------------
# 1) Yazarların kodunu değiştirmeden yükle
# ----------------------------------------------------------------------------------
def _load_module(name, path, package=None):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_authors_code():
    naim_dir = os.path.join(NAIM_REPO, "CMC_utils", "models", "tabular", "naim")
    pkg = types.ModuleType("authors_naim")
    pkg.__path__ = [naim_dir]
    sys.modules["authors_naim"] = pkg
    _load_module("authors_naim.tabular_tokenizer", os.path.join(naim_dir, "tabular_tokenizer.py"))
    naim = _load_module("authors_naim.naim", os.path.join(naim_dir, "naim.py"))
    aug = _load_module("authors_aug", os.path.join(NAIM_REPO, "CMC_utils", "data_augmentation", "tabular.py"))
    reg = _load_module("authors_reg", os.path.join(NAIM_REPO, "CMC_utils", "regularizers", "l1_l2_regularizers.py"))
    return naim.NAIM, aug.missing_augmentation, reg.L1RegularizationLoss, reg.L2RegularizationLoss


NAIM, missing_augmentation, L1Loss, L2Loss = load_authors_code()

# confs/experiment/model/naim.yaml
NAIM_PARAMS = dict(d_token=6, embedder_initialization="normal", bias=False, mask_type=2,
                   missing_value="~inf", num_heads=3, feedforward_dim=1000, dropout_rate=0.0,
                   activation="relu", num_layers=6, extractor=False)
# confs/experiment/model_type_params/dl_params.yaml + regularizer/l1.yaml, l2.yaml
DL_PARAMS = dict(max_epochs=1500, min_epochs=50, batch_size=32,
                 learning_rates=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7], early_stopping_patience=50,
                 scheduler_patience=25, performance_tolerance=1e-3, l1_alpha=1e-4, l2_alpha=1e-4)


# ----------------------------------------------------------------------------------
# 2) Veri ayrımı ve ek eksiklik (MCAR) üretimi
# ----------------------------------------------------------------------------------
def hide_observed(X: pd.DataFrame, ratio: float, rng: np.random.Generator) -> pd.DataFrame:
    """Gözlenen (NaN olmayan) hücrelerin `ratio` kadarını rastgele gizler (MCAR).
    Her hastada en az bir gözlenen değer bırakılır."""
    if ratio == 0:
        return X.copy()
    vals = X.values.astype(object)
    observed = np.argwhere(~X.isna().values)
    n_hide = int(round(ratio * len(observed)))
    hide = observed[rng.choice(len(observed), size=n_hide, replace=False)]
    mask = np.zeros(X.shape, dtype=bool)
    mask[hide[:, 0], hide[:, 1]] = True
    for r in np.where((X.isna().values | mask).all(axis=1))[0]:   # tamamen boşalan satır
        cols = np.where(mask[r])[0]
        mask[r, rng.choice(cols)] = False
    vals[mask] = np.nan
    return pd.DataFrame(vals, index=X.index, columns=X.columns).infer_objects()


def make_condition(df, split, ratio, seed):
    """Bir deney koşulu (oran, seed) için TÜM modellerin kullanacağı ortak veri."""
    out = {}
    for k, name in enumerate(["train", "val", "test"]):
        rng = np.random.default_rng([seed, int(round(ratio * 100)), k])
        X = hide_observed(df.loc[split[name], FEATURES], ratio, rng)
        out[name] = (X, encode_labels(df.loc[split[name], "class"]))
    return out


# ----------------------------------------------------------------------------------
# 3) NAIM eğitimi (yazarların TrainManager mantığı)
# ----------------------------------------------------------------------------------
class TabDataset(Dataset):
    """supervised_torch.SupervisedTabularDatasetTorch.__getitem__ ile aynı davranış."""

    def __init__(self, X, y, augmentation=False):
        self.X, self.y, self.augmentation = X.astype(float), y, augmentation

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        sample = np.array(self.X[i], dtype=float)
        if self.augmentation:
            sample = missing_augmentation(sample.copy())   # yazarların regularization'ı
        return torch.tensor(sample, dtype=torch.float32), int(self.y[i])


def init_weights(model):
    """models/miscellaneous.initialize_weights: Linear -> xavier_normal, bias 0."""
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight, gain=1.0)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)


@torch.no_grad()
def predict_proba_naim(model, X):
    model.eval()
    logits = model(torch.tensor(X, dtype=torch.float32))
    return F.softmax(logits, dim=-1)[:, 1].numpy()


def train_naim(cond, seed, log_every=0, stop_metric="auc"):
    """stop_metric='auc': yazarların kodu (val AUC ile early stopping + LR düşürme).
    stop_metric='loss': makale metni (Ek B) ile uyumlu varyant; val cross-entropy kaybı kullanılır."""
    torch.manual_seed(seed)
    np.random.seed(seed)           # missing_augmentation numpy global RNG kullanıyor

    Xtr_df, ytr = cond["train"]
    prep = NaimPreprocessor().fit(Xtr_df)
    Xtr, Xva = prep.transform(Xtr_df), prep.transform(cond["val"][0])
    yva = cond["val"][1]

    model = NAIM(input_size=len(FEATURES), output_size=len(CLASSES), cat_idxs=prep.cat_idxs,
                 cat_dims=prep.cat_dims, **NAIM_PARAMS)
    init_weights(model)

    p = DL_PARAMS
    weights = compute_sample_weight("balanced", ytr)
    g = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, num_samples=len(ytr), replacement=False, generator=g)
    loader = DataLoader(TabDataset(Xtr, ytr, augmentation=True), batch_size=p["batch_size"],
                        sampler=sampler, drop_last=True)

    l1, l2 = L1Loss(), L2Loss()
    lr_idx = 0
    opt = torch.optim.Adam(model.parameters(), lr=p["learning_rates"][lr_idx], betas=(0.9, 0.999), eps=1e-8)
    best = dict(perf=-np.inf, epoch=-1, val_auc=np.nan, val_loss=np.nan, weights=copy.deepcopy(model.state_dict()))
    es_no_improve = sch_no_improve = 0
    history = []

    for epoch in range(p["max_epochs"]):
        model.train()
        run_loss, n = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, F.one_hot(yb, 2).float())
            loss = loss + p["l1_alpha"] * l1(model) + p["l2_alpha"] * l2(model)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(yb)
            n += len(yb)

        pva = predict_proba_naim(model, Xva)
        val_auc = roc_auc_score(yva, pva)
        val_loss = float(F.binary_cross_entropy(torch.tensor(pva, dtype=torch.float64).clamp(1e-7, 1 - 1e-7),
                                                torch.tensor(yva, dtype=torch.float64)))
        history.append((epoch, run_loss / max(n, 1), val_auc, val_loss, p["learning_rates"][lr_idx]))

        # --- TrainManager.callbacks ---
        score = val_auc if stop_metric == "auc" else -val_loss      # büyük olan daha iyi
        improved = epoch > p["min_epochs"] and (score - best["perf"]) > p["performance_tolerance"]
        if improved:
            best.update(perf=score, epoch=epoch, val_auc=val_auc, val_loss=val_loss,
                        weights=copy.deepcopy(model.state_dict()))
            es_no_improve = sch_no_improve = 0
        else:
            if epoch > p["min_epochs"]:
                es_no_improve += 1
                sch_no_improve += 1
                if sch_no_improve >= p["scheduler_patience"] and lr_idx < len(p["learning_rates"]) - 1:
                    lr_idx += 1
                    opt = torch.optim.Adam(model.parameters(), lr=p["learning_rates"][lr_idx],
                                           betas=(0.9, 0.999), eps=1e-8)
                    sch_no_improve = 0
            if es_no_improve >= p["early_stopping_patience"]:
                break
        if log_every and epoch % log_every == 0:
            print(f"  epoch {epoch:4d} loss {run_loss / max(n, 1):.4f} val_auc {val_auc:.4f}", flush=True)

    model.load_state_dict(best["weights"])
    info = dict(best_epoch=best["epoch"], best_val_auc=best["val_auc"], best_val_loss=best["val_loss"], last_epoch=epoch)
    return model, prep, info, history


# ----------------------------------------------------------------------------------
# 4) Karşılaştırma modelleri (repodaki svm.yaml / rf.yaml ayarları; KNN sklearn varsayılanı)
# ----------------------------------------------------------------------------------
def baseline_models(seed):
    return {
        "SVM": SVC(C=1.0, kernel="rbf", gamma="scale", probability=True,
                   class_weight="balanced", random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "RF": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed),
    }


def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    rec = recall_score(y, pred, average=None, labels=[0, 1], zero_division=0)
    return dict(accuracy=accuracy_score(y, pred), macro_f1=f1_score(y, pred, average="macro"),
                recall_notckd=rec[0], recall_ckd=rec[1], auc=roc_auc_score(y, prob))


def save(args, rows, preds, histories):
    pd.DataFrame(rows).to_csv(os.path.join(args.out, f"runs{args.tag}.csv"), index=False)
    pd.DataFrame(preds).to_csv(os.path.join(args.out, f"test_predictions{args.tag}.csv"), index=False)
    pd.DataFrame(histories).to_csv(os.path.join(args.out, f"naim_training_history{args.tag}.csv"), index=False)


# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "chronic_kidney_disease_full.arff"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--ratios", type=float, nargs="*", default=MISSING_RATIOS)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--stop-metric", choices=["auc", "loss"], default="auc",
                    help="NAIM early stopping/LR ölçütü: auc (yazarların kodu) veya loss (makale metni)")
    ap.add_argument("--naim-only", action="store_true", help="Sadece NAIM'i eğit (baseline'ları atla)")
    ap.add_argument("--tag", default="", help="Çıktı dosya adlarına eklenecek etiket, ör. _valloss")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    df = load_ckd(args.data)
    split = fixed_split(df)
    with open(os.path.join(args.out, "split.json"), "w") as f:
        json.dump({k: [int(i) for i in v] for k, v in split.items()}, f)

    rows, preds, histories = [], [], []
    for ratio in args.ratios:
        for seed in args.seeds:
            cond = make_condition(df, split, ratio, seed)
            Xte_df, yte = cond["test"]
            miss = {k: float(v[0].isna().values.mean()) for k, v in cond.items()}
            base = dict(ratio=ratio, seed=seed, **{f"missing_{k}": v for k, v in miss.items()})

            t0 = time.time()
            model, prep, info, hist = train_naim(cond, seed, stop_metric=args.stop_metric)
            prob = predict_proba_naim(model, prep.transform(Xte_df))
            naim_name = "NAIM" if args.stop_metric == "auc" else "NAIM (val-loss)"
            rows.append(dict(base, model=naim_name, **metrics(yte, prob), **info, seconds=time.time() - t0))
            preds += [dict(ratio=ratio, seed=seed, model=naim_name, patient_id=int(i), y=int(t), p=float(q))
                      for i, t, q in zip(Xte_df.index, yte, prob)]
            histories += [dict(ratio=ratio, seed=seed, epoch=e, train_loss=l, val_auc=a, val_loss=vl, lr=lr) for e, l, a, vl, lr in hist]
            print(f"[ratio {ratio:.2f} seed {seed}] {naim_name}  acc={rows[-1]['accuracy']:.3f} "
                  f"f1={rows[-1]['macro_f1']:.3f} best_epoch={info['best_epoch']} ({rows[-1]['seconds']:.0f}s)", flush=True)

            if args.naim_only:
                save(args, rows, preds, histories)
                continue
            bprep = BaselinePreprocessor().fit(cond["train"][0])
            Xtr, Xte = bprep.transform(cond["train"][0]), bprep.transform(Xte_df)
            for name, clf in baseline_models(seed).items():
                clf.fit(Xtr, cond["train"][1])
                prob = clf.predict_proba(Xte)[:, 1]
                rows.append(dict(base, model=name, **metrics(yte, prob)))
                preds += [dict(ratio=ratio, seed=seed, model=name, patient_id=int(i), y=int(t), p=float(q))
                          for i, t, q in zip(Xte_df.index, yte, prob)]
                print(f"[ratio {ratio:.2f} seed {seed}] {name:4s}  acc={rows[-1]['accuracy']:.3f} "
                      f"f1={rows[-1]['macro_f1']:.3f}", flush=True)

            save(args, rows, preds, histories)

    print("Bitti. Sonuçlar:", args.out)


if __name__ == "__main__":
    main()

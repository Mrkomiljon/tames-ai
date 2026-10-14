"""Fill the result placeholders of the manuscript from a run directory.
usage: python fill_results.py <run_dir> <processed_dir> <manuscript.md> <response_letter.md|-> [--write]
Without --write only prints the replacements."""
import json, sys, os, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from tames.eval import bootstrap_ci, classification_metrics, calibration_metrics

run, proc, man_path, resp_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
WRITE = "--write" in sys.argv
R = json.load(open(run / "ensemble_tames_seed0/results.json"))
M = R["models"]; C = R["comparisons_vs_tames"]
abl = pd.read_csv(run / "ablation_table.csv")
shap = pd.read_csv(run / "explain_tames_seed0/shap_summary.csv", index_col=0)
stab = json.load(open(run / "explain_tames_seed0/shap_stability.json"))
faith = json.load(open(run / "explain_tames_seed0/faithfulness.json"))
order = json.load(open(run / "ensemble_tames_seed0/model_order.json"))
P = np.load(run / "ensemble_tames_seed0/p_pooled_test.npy")
meta_te = pd.read_csv(proc / "meta_test.csv"); meta_tr = pd.read_csv(proc / "meta_train.csv")
y = meta_te["label"].values.astype(int)
n_boot = 2000

f3 = lambda v: f"{v:.3f}"
pct = lambda v: f"{100 * v:.1f}"
def ci3(ci): return f"\\[{ci[0]:.3f}, {ci[1]:.3f}\\]"
def cip(ci): return f"\\[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}\\]"
def pfmt(p): return "< 0.001" if p < 0.001 else f"{p:.3f}"
def signed(v): return f"{v:+.3f}"

HUMAN = {"taller_than_wide": "taller-than-wide ratio", "solidity": "solidity", "eccentricity": "eccentricity",
         "irregularity": "margin irregularity", "bright_frac": "bright-spot fraction", "echo_std": "echo SD",
         "area_frac": "area fraction", "echo_mean": "echo mean", "echo_ratio": "echo ratio", "echo_entropy": "echo entropy",
         "n_components": "component count", "Q_img": "Q\\_img", "Q_struct": "Q\\_struct",
         "image embedding (Σ|φ| over h_img)": "the image embedding h\\_img (summed over its 128 dimensions)"}
hum = lambda k: HUMAN.get(k, k)

T = man_path.read_text()
log = []
def rep(old, new, count=1):
    global T
    assert T.count(old) == count, f"{T.count(old)} occurrences (expected {count}) of: {old[:100]!r}"
    T = T.replace(old, new); log.append((old[:70], new[:110]))

# ---- Table 3 ------------------------------------------------------------
tm, lf = M["tames"], M["late_fusion_cnn_xgb"]
rows = {"Image-only CNN": "cnn_image_only", "Descriptor-only XGBoost": "xgb_struct_only",
        "CNN + XGBoost late fusion": "late_fusion_cnn_xgb", "DL learner alone": "dl_only", "RF learner alone": "rf_only",
        "XGB learner alone": "xgb_only", "FSVM learner alone": "fsvm_only", "TAMES AI": "tames"}
for name, key in rows.items():
    m, t, ci = M[key], M[key]["test"], M[key]["test_ci"]
    rep(f"| {name} | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] | [[ ]] |",
        f"| {name} | {f3(t['auc'])} {ci3(ci['auc'])} | {pct(t['sensitivity'])} {cip(ci['sensitivity'])} | {pct(t['specificity'])} {cip(ci['specificity'])} | "
        f"{pct(t['accuracy'])} {cip(ci['accuracy'])} | {pct(t['f1'])} {cip(ci['f1'])} | {f3(t['mcc'])} {ci3(ci['mcc'])} |")
rep("discrimination and metrics at the pre-specified threshold t\\* = [[ ]]; 95 % bootstrap CIs in brackets.",
    f"discrimination and metrics at each model's pre-specified threshold t\\* (sensitivity ≥ 0.95 on validation; TAMES AI t\\* = {f3(tm['threshold']['t_star'])}, Youden threshold {f3(tm['threshold']['t_youden'])}); 95 % bootstrap CIs in brackets. Metrics are computed on test probabilities averaged over the five seeds.")
d = C["late_fusion_cnn_xgb"]; dl, ms, mp = d["delong"], d["mcnemar_sensitivity"], d["mcnemar_specificity"]
sens_d = 100 * (tm["test"]["sensitivity"] - lf["test"]["sensitivity"]); spec_d = 100 * (tm["test"]["specificity"] - lf["test"]["specificity"])
rep("The AUC difference between TAMES AI and the CNN + XGBoost late-fusion baseline was [[ ]] [[CI]] (DeLong z = [[ ]], p = [[ ]], Holm-corrected p = [[ ]]). At t\\*, sensitivity differed by [[ ]] percentage points and specificity by [[ ]] points; exact McNemar tests on discordant pairs ([[b]] vs [[c]] for sensitivity, [[b]] vs [[c]] for specificity) gave p = [[ ]] and p = [[ ]].",
    f"The AUC difference between TAMES AI and the CNN + XGBoost late-fusion baseline was {signed(dl['diff'])} {ci3((dl['ci_low'], dl['ci_high']))} (DeLong z = {dl['z']:.2f}, p = {pfmt(dl['p'])}, Holm-corrected p = {pfmt(dl['p_holm'])}). At t\\*, sensitivity differed by {sens_d:+.1f} percentage points and specificity by {spec_d:+.1f} points; exact McNemar tests on discordant pairs ({ms['b_A_right_B_wrong']} vs {ms['c_B_right_A_wrong']} for sensitivity, {mp['b_A_right_B_wrong']} vs {mp['c_B_right_A_wrong']} for specificity) gave p = {pfmt(ms['p'])} and p = {pfmt(mp['p'])}.")
rep("Seed-to-seed standard deviation of the TAMES AI test AUC was [[ ]]", f"Seed-to-seed standard deviation of the TAMES AI test AUC was {f3(tm['test_auc_seed_sd'])} (per-seed mean {f3(tm['test_auc_seed_mean'])})")
# all comparisons as one sentence
others = {"cnn_image_only": "the image-only CNN", "xgb_struct_only": "the descriptor-only XGBoost", "dl_only": "the DL learner alone",
          "rf_only": "the RF learner alone", "xgb_only": "the XGB learner alone", "fsvm_only": "the FSVM learner alone"}
parts = [f"{nm} {signed(C[k]['delong']['diff'])} (Holm p = {pfmt(C[k]['delong']['p_holm'])})" for k, nm in others.items() if k in C]
rep("Figure 3 shows the ROC curves of all models.", "AUC differences of TAMES AI against the other baselines were: " + "; ".join(parts) + ". Figure 3 shows the ROC curves of all models.")

# ---- Abstract -----------------------------------------------------------
rep("TAMES AI reached an AUC of [[ ]] [[CI]], sensitivity [[ ]] % [[CI]], specificity [[ ]] % [[CI]] and accuracy [[ ]] % at the pre-specified threshold, against an AUC of [[ ]] [[CI]] for CNN + XGBoost late fusion (DeLong p = [[ ]], Holm-corrected p = [[ ]]).",
    f"TAMES AI reached an AUC of {f3(tm['test']['auc'])} {ci3(tm['test_ci']['auc'])}, sensitivity {pct(tm['test']['sensitivity'])} % {cip(tm['test_ci']['sensitivity'])}, specificity {pct(tm['test']['specificity'])} % {cip(tm['test_ci']['specificity'])} and accuracy {pct(tm['test']['accuracy'])} % at the pre-specified threshold, against an AUC of {f3(lf['test']['auc'])} {ci3(lf['test_ci']['auc'])} for CNN + XGBoost late fusion (DeLong p = {pfmt(dl['p'])}, Holm-corrected p = {pfmt(dl['p_holm'])}).")

# ---- Ablation -----------------------------------------------------------
amap = {"(a) AFMO off (F → μ)": "| (a) AFMO (F replaced by μ) |", "(b) ATCFM off (plain Gaussian)": "| (b) ATCFM (plain Gaussian membership) |",
        "(c) quality coefficients off (Q_m = 1)": "| (c) Quality coefficients (Q\\_m = 1) |", "(e) equal ensemble weights (τ → ∞)": "| (e) Adaptive weights (equal ω\\_k) |",
        "(f) without DL learner": "| (f1) DL learner |", "(f) without RF learner": "| (f2) RF learner |",
        "(f) without XGB learner": "| (f3) XGB learner |", "(f) without FSVM learner": "| (f4) FSVM learner |"}
labels = {"(a) AFMO off (F → μ)": "AFMO (a)", "(b) ATCFM off (plain Gaussian)": "ATCFM (b)", "(c) quality coefficients off (Q_m = 1)": "the quality coefficients (c)",
          "(e) equal ensemble weights (τ → ∞)": "the adaptive weights (e)", "(f) without DL learner": "the DL learner (f1)", "(f) without RF learner": "the RF learner (f2)",
          "(f) without XGB learner": "the XGB learner (f3)", "(f) without FSVM learner": "the FSVM learner (f4)"}
A = abl.set_index("ablation")
for k, row in amap.items():
    r = A.loc[k]
    rep(f"{row} [[ ]] | [[ ]] |", f"{row} {f3(r['test_auc'])} | {signed(r['delta_auc'])} {ci3((r['ci_low'], r['ci_high']))}, DeLong p = {pfmt(r['delong_p'])} |")
rep("| None (full TAMES AI) | [[ ]] | — |", f"| None (full TAMES AI) | {f3(tm['test']['auc'])} {ci3(tm['test_ci']['auc'])} | — |")
sub = A.loc[list(amap)]
worst = sub["delta_auc"].idxmin(); w = sub.loc[worst]
within = [labels[k] for k in sub.index if sub.loc[k, "ci_low"] <= 0 <= sub.loc[k, "ci_high"]]
worst_txt = f"{labels[worst]} (ΔAUC {signed(w['delta_auc'])} {ci3((w['ci_low'], w['ci_high']))})"
rep("The largest loss followed removal of [[component]] ([[ΔAUC]]); removal of [[component]] did not change AUC beyond its confidence interval.",
    f"The largest loss followed removal of {worst_txt}; removal of {', '.join(within) if within else 'no component'} did not change AUC beyond the confidence interval of the difference.")
W = pd.DataFrame([ps["weights"] for ps in R["per_seed"]]); taus = [ps["tau"] for ps in R["per_seed"]]
tau_txt = f"τ = {taus[0]} in all five seeds" if len(set(taus)) == 1 else "τ = " + ", ".join(str(t) for t in taus) + " (seeds 0–4)"
rep("The learned ensemble weights were ω\\_DL = [[ ]], ω\\_RF = [[ ]], ω\\_XGB = [[ ]], ω\\_FSVM = [[ ]] at τ = [[ ]].",
    f"The learned ensemble weights, averaged over the five seeds (± SD), were ω\\_DL = {W['dl'].mean():.3f} ± {W['dl'].std(ddof=0):.3f}, ω\\_RF = {W['rf'].mean():.3f} ± {W['rf'].std(ddof=0):.3f}, ω\\_XGB = {W['xgb'].mean():.3f} ± {W['xgb'].std(ddof=0):.3f}, ω\\_FSVM = {W['fsvm'].mean():.3f} ± {W['fsvm'].std(ddof=0):.3f} at {tau_txt}.")
abs_txt = f"removal of {labels[worst]} cost the most AUC ({signed(w['delta_auc'])} {ci3((w['ci_low'], w['ci_high']))})"
rep("Ablation showed that [[component]] contributed most ([[ΔAUC]]).", f"In the ablation, {abs_txt}.")

# ---- Calibration (CIs for slope/intercept by bootstrap of the pooled test predictions) ----
cal = tm["calibration_test"]; p_t = P[order.index("tames")]
rng = np.random.default_rng(0); sl, ic = [], []
for _ in range(n_boot):
    idx = rng.integers(0, len(y), len(y))
    if len(np.unique(y[idx])) < 2: continue
    c = calibration_metrics(y[idx], p_t[idx]); sl.append(c["slope"]); ic.append(c["intercept"])
rep("On the internal test partition the Brier score was [[ ]], the calibration slope [[ ]] [[CI]] and the intercept [[ ]] [[CI]].",
    f"On the internal test partition the Brier score was {f3(cal['brier'])}, the calibration slope {cal['slope']:.2f} \\[{np.percentile(sl, 2.5):.2f}, {np.percentile(sl, 97.5):.2f}\\] and the intercept {cal['intercept']:.2f} \\[{np.percentile(ic, 2.5):.2f}, {np.percentile(ic, 97.5):.2f}\\] (slope and intercept CIs from {n_boot} bootstrap resamples of the pooled test predictions).")

# ---- Subgroups by size ---------------------------------------------------
med = float(meta_tr["area_frac"].median()); t_star = tm["threshold"]["t_star"]
for name, mask in (("| Smaller nodules (area fraction < 0.082) |", meta_te["area_frac"].values < med), ("| Larger nodules (area fraction ≥ 0.082) |", meta_te["area_frac"].values >= med)):
    ys, ps = y[mask], p_t[mask]
    m = classification_metrics(ys, ps, t_star); ci = bootstrap_ci(ys, ps, t_star, n_boot=n_boot, seed=0)
    rep(f"{name} [[ ]] | [[ ]] | [[ ]] | [[ ]] |", f"{name} {int(mask.sum())} ({int(ys.sum())} malignant) | {f3(m['auc'])} {ci3(ci['auc'])} | {pct(m['sensitivity'])} {cip(ci['sensitivity'])} | {pct(m['specificity'])} {cip(ci['specificity'])} |")

# ---- Explanations --------------------------------------------------------
top = list(shap["ensemble"].sort_values(ascending=False).index[:10])
freq = stab["in_top10_frequency"]; lo = min(freq, key=freq.get)
n_ge = sum(v >= 0.95 for v in freq.values())
rep("were, in order, [[top-10 SHAP features]]; [[rank stability]] (Figure 5a).",
    f"were, in order, {', '.join(hum(k) for k in top)} (ensemble attribution of {stab['n_rows']} test images, Supplementary Table S5); {n_ge} of the ten stayed in the top ten in at least 95 % of {stab['n_boot']} bootstrap resamples, the least stable being {hum(lo)} ({100 * freq[lo]:.0f} %) (Figure 5a).")
rep("Grad-CAM deletion and insertion AUCs were [[ ]] and [[ ]], against [[ ]] and [[ ]] for a random-saliency control (Figure 5b).",
    f"Grad-CAM deletion and insertion AUCs on {faith['n_images']} test images were {f3(faith['deletion_auc_gradcam'])} and {f3(faith['insertion_auc_gradcam'])}, against {f3(faith['deletion_auc_random'])} and {f3(faith['insertion_auc_random'])} for a random-saliency control (lower deletion and higher insertion AUC indicate a faithful map) (Figure 5b).")
faithful = faith["deletion_auc_gradcam"] < faith["deletion_auc_random"] and faith["insertion_auc_gradcam"] > faith["insertion_auc_random"]
rep("[[dominant SHAP features]] were the dominant SHAP features.", f"{', '.join(hum(k) for k in top[:5])} were the dominant SHAP features.")

# ---- Verdict phrases -----------------------------------------------------
auc_t = tm["test"]["auc"]; better_all = all(auc_t > M[k]["test"]["auc"] for k in ("cnn_image_only", "xgb_struct_only", "late_fusion_cnn_xgb"))
sig = dl["p_holm"] < 0.05 and dl["diff"] > 0
if better_all and sig: verdict, concl, disc = "improved", "outperformed", "the gain"
elif better_all: verdict, concl, disc = "numerically but not significantly improved", "numerically but not significantly outperformed", "the largest change"
elif dl["diff"] > 0: verdict, concl, disc = "did not consistently improve", "did not consistently outperform", "the largest change"
else: verdict, concl, disc = "did not improve", "did not outperform", "the largest change"
rep("Its image-plus-descriptor instantiation [[improved / did not improve]] discrimination", f"Its image-plus-descriptor instantiation {verdict} discrimination")
rep("On TN3K, TAMES AI [[improved / did not improve]] discrimination", f"On TN3K, TAMES AI {verdict} discrimination")
rep("and the ablation locates [[the gain / the largest change]] in [[component(s)]] (Table 4).", f"and the ablation locates {disc} in {labels[worst]} (Table 4).")
rep("produced a thyroid-nodule classifier that [[outperformed / matched / did not outperform]] single-modality", f"produced a thyroid-nodule classifier that {concl} single-modality")
pos = "below" if auc_t < 0.90 else "within"
rep("the AUC reported here ([[ ]]) is [[below / within / above]] the pooled range;", f"the AUC reported here ({f3(auc_t)}) is {pos} the pooled range of those image-only classifiers (above 0.90);")
rep("Grad-CAM maps were [[more / not more]] faithful than a random-saliency control, and the SHAP ranking was [[stable / partly stable]] under resampling",
    f"Grad-CAM maps were {'more' if faithful else 'not more'} faithful than a random-saliency control, and the SHAP ranking was {'stable' if n_ge == 10 else 'partly stable'} under resampling")

# ---- training time from model.pt timestamps -------------------------------
ts = [os.path.getmtime(run / f"cnn_tames_seed{s}/model.pt") for s in range(5) if (run / f"cnn_tames_seed{s}/model.pt").exists()]
dur = [(ts[i] - ts[i - 1]) / 60 for i in range(1, len(ts))]
if dur:
    ep = [len(json.load(open(run / f"cnn_tames_seed{s}/cnn_results.json"))["history"]) for s in range(len(ts))]
    rep("training time [[h]] per seed", f"backbone training {min(dur):.1f}–{max(dur):.1f} min per seed ({min(ep)}–{max(ep)} epochs; from output timestamps)")

print("\n".join(f"- {o}\n    → {n}" for o, n in log)); print("remaining [[:", T.count("[["))
if WRITE:
    man_path.write_text(T)
    if resp_path != "-":
        rp = Path(resp_path); Rt = rp.read_text(); Rt = Rt.replace(" [[fill from results]]", ""); rp.write_text(Rt)
    print("written")

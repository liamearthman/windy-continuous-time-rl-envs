"""
Plot the ct_sac 5-seed control results: per-seed success rate and final error,
with the aggregate mean +/- std overlaid.

Run in VS Code:  open this file and hit "Run Python File" (or use the play button),
or from a terminal:   .venv/bin/python results/plot_ct_sac_seeds.py
"""
import glob
import json
import statistics as st

import matplotlib.pyplot as plt

# --- load per-seed results -------------------------------------------------
files = sorted(glob.glob("results/ct_sac_seed*.json"))
seeds, succ, ferr = [], [], []
for f in files:
    d = json.load(open(f))
    seeds.append(int(f.split("seed")[-1].split(".")[0]))
    succ.append(d["success_rate"] * 100.0)   # to percent
    ferr.append(d["final_error_mean"])

succ_m, succ_s = st.mean(succ), st.pstdev(succ)
ferr_m, ferr_s = st.mean(ferr), st.pstdev(ferr)

# --- plot ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
x = range(len(seeds))

# Success rate
ax1.bar(x, succ, color="#4C72B0", alpha=0.85)
ax1.axhline(succ_m, color="crimson", ls="--", lw=1.5, label=f"mean {succ_m:.1f}%")
ax1.axhspan(succ_m - succ_s, succ_m + succ_s, color="crimson", alpha=0.12,
            label=f"±std {succ_s:.1f}")
ax1.set_xticks(list(x)); ax1.set_xticklabels([f"s{s}" for s in seeds])
ax1.set_ylim(0, 105); ax1.set_ylabel("Success rate (%)")
ax1.set_title("ct_sac success rate by seed"); ax1.legend(loc="lower right")

# Final error
ax2.bar(x, ferr, color="#55A868", alpha=0.85)
ax2.axhline(ferr_m, color="crimson", ls="--", lw=1.5, label=f"mean {ferr_m:.3f}")
ax2.axhspan(ferr_m - ferr_s, ferr_m + ferr_s, color="crimson", alpha=0.12,
            label=f"±std {ferr_s:.3f}")
ax2.axhline(0.1, color="black", ls=":", lw=1.2, label="ε = 0.1 (success ball)")
ax2.set_xticks(list(x)); ax2.set_xticklabels([f"s{s}" for s in seeds])
ax2.set_ylabel("Final error (rad)")
ax2.set_title("ct_sac final error by seed"); ax2.legend(loc="upper right")

fig.suptitle("ct_sac pendulum — 5-seed control results (uniform time, matched plant)",
             fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])

out = "results/ct_sac_5seed.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")
print(f"success: {succ_m:.1f}% ± {succ_s:.1f}   final_error: {ferr_m:.4f} ± {ferr_s:.4f}")
plt.show()

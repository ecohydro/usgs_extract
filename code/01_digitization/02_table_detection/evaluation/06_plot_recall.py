"""Figure: share of listed table pages the detector flags, against the confidence threshold.
Reads OUT/recall_curve.csv; writes OUT/figures/recall_by_threshold.png and .pdf. Run: python 06_plot_recall.py"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import defaults as D


def main():
    c = pd.read_csv(D.OUT / "recall_curve.csv")
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(3.4, 2.8), constrained_layout=True)
    a, b = c[c.list_version == "cleaned"], c[c.list_version == "as_listed"]
    ax.fill_between(a.threshold, a.low, a.high, color="#31688e", alpha=0.25, lw=0)
    ax.plot(a.threshold, a.recall, color="#31688e", lw=1.6, label="list checked by eye")
    ax.plot(b.threshold, b.recall, color="0.6", lw=1, label="list as given")
    for t in D.THRESHOLDS_USED:
        r = a[a.threshold == t].recall.iloc[0]
        ax.axvline(t, color="k", ls=":", lw=0.8)
        left = t == min(D.THRESHOLDS_USED)
        ax.annotate(f"{r:.0%}", (t, r), xytext=(-4, 5) if left else (4, -11), textcoords="offset points",
                    ha="right" if left else "left", fontsize=8)
    ax.set(xlabel="confidence threshold", ylabel="share of table pages flagged", xlim=(0.05, 1), ylim=(0, 1),
           title="Table pages found, by threshold")
    ax.legend(frameon=False, loc="lower left")
    (D.OUT / "figures").mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(D.OUT / "figures" / f"recall_by_threshold.{ext}", dpi=200)


if __name__ == "__main__":
    main()

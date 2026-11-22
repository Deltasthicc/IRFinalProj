"""Draw the answer-checking flow used in the report: python docs/report/figures/make_flow.py -> fig_answer_flow.png"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "fig_answer_flow.png"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
fig, ax = plt.subplots(figsize=(6.5, 2.7))
ax.set_xlim(0, 100)
ax.set_ylim(0, 42)
ax.axis("off")


def box(x, y, w, h, title, body="", fc="#e8f1fb", ec="#2b5c8a"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="top", fontweight="bold", fontsize=7.5)
    if body:
        ax.text(x + w / 2, y + h - 8.5, body, ha="center", va="top", fontsize=6.6, linespacing=1.35)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=1.1, color="#333"))


box(1, 18, 21, 20, "Question", 'e.g. "How does Telsa\ndescribe CEO risk?"\n(the typo is fixed first)')
box(26, 18, 21, 20, "Best 5 passages", "from hybrid search\nand the reranker,\nlabelled [P1] to [P5]", fc="#eaf6ee", ec="#2f7d4f")
box(51, 18, 22, 20, "Language model", "answers only from the\n5 passages and cites\nevery claim (temp. 0)", fc="#fff4e0", ec="#b36b00")
box(77, 18, 22, 20, "Checker (no AI)", "1. cited [Pn] exists\n2. every sentence cited\n3. every number is in\n    a cited passage", fc="#f3e8fa", ec="#6a3d8f")
arrow(22.6, 28, 25.4, 28)
arrow(47.6, 28, 50.4, 28)
arrow(73.6, 28, 76.4, 28)
box(51, 0, 22, 13, "Decline", '"insufficient evidence"', fc="#fdeaea", ec="#a33")
box(77, 0, 22, 13, "Show the answer", "with a trust panel\n(each check: pass or flag)", fc="#eaf6ee", ec="#2f7d4f")
arrow(62, 17.4, 62, 13.6)
ax.text(63, 14.9, "model says it is unsure", fontsize=6.5, ha="left")
arrow(88, 17.4, 88, 13.6)
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("wrote", OUT.name)

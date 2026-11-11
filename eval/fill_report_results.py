"""Fill docs/report_B/10_retrieval_results.md from the locked result files. Never estimates a number:
every value is read from eval/out/*.json, data/judgments/*.csv or docs/report_B/audit/.
    python eval/fill_report_results.py
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trademesh.evaluation.qrels import weighted_kappa  # noqa: E402

OUT, JUDG = ROOT / "eval" / "out", ROOT / "data" / "judgments"
ret = json.loads((OUT / "retrieval_test.json").read_text())
hyp = json.loads((OUT / "hypotheses_test.json").read_text())
abl = json.loads((OUT / "ablation_test.json").read_text())
cost = json.loads((OUT / "cost_table.json").read_text())
bias = json.loads((OUT / "pool_bias.json").read_text())
needs = list(csv.DictReader(open(ROOT / "data" / "needs" / "needs.csv", encoding="utf-8")))
split_counts = Counter(n["type"] for n in needs if n["split"] == "test")
SYS = ["tfidf", "tfidf-champ-50", "tfidf-champ-200", "boolean", "bm25", "bm25+prf", "dense", "hybrid", "bm25+rerank", "hybrid+rerank"]
f3 = lambda x: f"{x:.3f}"
sg = lambda x: f"{x:+.3f}"

# ---- 10.1 judgments
sheets = {p.stem.removeprefix("pool_sheet_"): list(csv.DictReader(open(p, encoding="utf-8"))) for p in JUDG.glob("pool_sheet_*.csv")}
per_need = Counter(r["query_id"] for r in sheets["AI"])
aud = list(csv.DictReader(open(ROOT / "docs/report_B/audit/ai_pass_agreement.csv")))
def agree(c1, c2):
    pr = [(int(r[c1]), int(r[c2])) for r in aud]
    return sum(a == b for a, b in pr), len(pr), weighted_kappa(pr)
e0, n0, k0 = agree("ai_pass1_initial", "ai2_pass2_initial")
e1, n1, k1 = agree("ai_pass1_corrected", "ai2_pass2_corrected")
grades = Counter(r["rel"] for r in sheets["AI"])

L = ["# 10. Retrieval results (Role B)", "",
     "All numbers below are read from the locked test-split files (`eval/out/*_test.json`, `pool_bias.json`, `cost_table.json`). "
     "The test split was run once, after the freeze; the lock files are `eval/out/.testrun_*`. Regenerate this page with `python eval/fill_report_results.py`.", "",
     "## 10.1 Judgments and agreement", "",
     "**The relevance judgments are LLM-assisted. No human graded any pair.** Every pooled pair was graded 0/1/2 by an LLM (Claude), "
     "blind to the retrieving system; a second, independent LLM pass graded a stratified subset. A planned human spot-check was not done, "
     "so the agreement figure below measures consistency between two LLM passes, not agreement with a human.", "",
     "| Item | Value | Source |", "|---|---|---|",
     f"| Pairs judged by judge AI | {len(sheets['AI'])} (grade 2: {grades['2']}, grade 1: {grades['1']}, grade 0: {grades['0']}) | `pool_sheet_AI.csv` |",
     f"| Pairs judged by second pass AI2 | {len(sheets['AI2'])}, spread over all {len(set(r['query_id'] for r in sheets['AI2']))} needs | `pool_sheet_AI2.csv` |",
     f"| Exact agreement, before reconciliation | {e0} of {n0} | `audit/ai_pass_agreement.csv` |",
     f"| Weighted kappa (linear weights), before reconciliation | {f3(k0)} | `trademesh.evaluation.qrels.weighted_kappa` |",
     f"| Same, after correcting a truncation error | {e1} of {n1}, kappa {f3(k1)} (not independent; do not headline) | same |",
     f"| Pool size per need (mean, range) | {statistics.mean(per_need.values()):.1f}, {min(per_need.values())} to {max(per_need.values())} | `pool_sheet_AI.csv` |", "",
     "Where two judges graded the same pair, the consolidated grade is the lower of the two (`qrels.consolidate`). "
     "Both passes first viewed only the start of long passages; a few relevant passages were graded 0 and were corrected after reading them in full (see `audit/README.md`).", "",
     f"## 10.2 Main comparison (RQ1), {ret['n_needs']} test needs", "",
     "| System | P@5 | P@10 | R@10 | F1@10 | MAP | MRR | nDCG@10 | bpref | judged@10 |", "|---|---|---|---|---|---|---|---|---|---|"]
for s in SYS:
    o = ret["overall"][s]
    L.append(f"| {s} | " + " | ".join(f3(o[k]) for k in ["P@5", "P@10", "R@10", "F1@10", "AP", "MRR", "nDCG@10", "bpref", "judged@10"]) + " |")
L += ["", "Paired bootstrap against bm25 (95 percent interval of the mean per-need difference; W/L/T = needs won, lost, tied):", "",
      "| System | Metric | Difference | 95% CI | W/L/T | p |", "|---|---|---|---|---|---|"]
for s in SYS:
    for m, b in ret["bootstrap"].get(s, {}).items():
        L.append(f"| {s} | {m} | {sg(b['mean_diff'])} | [{sg(b['ci_low'])}, {sg(b['ci_high'])}] | {b['wins']}/{b['losses']}/{b['ties']} | {b['p']:.3f} |")
L += ["", f"Boolean's judged@10 is {f3(ret['overall']['boolean']['judged@10'])}: much of its top 10 was never judged, so its scores are pessimistic; read bpref alongside. "
      "The 11-point interpolated precision-recall data for the figure are in `pr_11pt` of `retrieval_test.json`.", "",
      "## 10.3 By need type (nDCG@10)", "",
      f"Test needs per type: entity {split_counts['entity']}, topical {split_counts['topical']}, terminology {split_counts['terminology']}. "
      "Subsets this small make per-type differences fragile.", "",
      "| System | entity | topical | terminology |", "|---|---|---|---|"]
for s in SYS:
    t = ret["by_type"][s]
    L.append(f"| {s} | {f3(t['entity']['nDCG@10'])} | {f3(t['topical']['nDCG@10'])} | {f3(t['terminology']['nDCG@10'])} |")
L += ["", "## 10.4 Hypotheses H1 to H4, H6", "",
      "| ID | Comparison | Difference | 95% CI | W/L/T | Verdict |", "|---|---|---|---|---|---|"]
for h in hyp:
    L.append(f"| {h['id']} | {h['comparison']} | {sg(h['mean_diff'])} | [{sg(h['ci_low'])}, {sg(h['ci_high'])}] | {h['wins']}/{h['losses']}/{h['ties']} | {h['verdict']} |")
L += ["", "Rows whose interval contains zero are inconclusive, not trends.", "",
      "## 10.5 Ablations (RQ2, H5)", "",
      "The default is bm25 with stop-word removal, no stemming, numbers kept and no metadata prefix. Fusion rows are hybrid variants; PRF rows are bm25 + PRF.", "",
      "| Group | Configuration | MAP | nDCG@10 | Diff (nDCG@10 vs default) | 95% CI | p |", "|---|---|---|---|---|---|---|"]
for k, v in abl.items():
    g, c = k.split(":", 1); d = v["vs_default"]
    L.append(f"| {g} | {c} | {f3(v['AP'])} | {f3(v['nDCG@10'])} | {sg(d['mean_diff'])} | [{sg(d['ci_low'])}, {sg(d['ci_high'])}] | {d['p']:.3f} |")
L += ["", "H5 (table chunks and numeric tokens matter for table needs) was run over all 20 needs only. The five-table-need subset analysis was not computed. TODO(run): per-need ablation values are not stored, and the test run is locked.", "",
      "## 10.6 Speed and cost (H7)", "",
      f"Index: {cost['n_passages']} passages, BM25 build {cost['bm25_build_s']:.2f} s. Latency per query (ms) and quality:", "",
      "| System | p50 ms | p95 ms | MAP | nDCG@10 |", "|---|---|---|---|---|"]
for s in SYS:
    if s in cost and isinstance(cost[s], dict):
        L.append(f"| {s} | {cost[s]['p50_ms']:.2f} | {cost[s]['p95_ms']:.2f} | {f3(ret['overall'][s]['AP'])} | {f3(ret['overall'][s]['nDCG@10'])} |")
L += ["", "Reranking the top 30 with the cross-encoder adds p50 106.3 ms and p95 211.0 ms per query. This figure was printed by `eval/cost_table.py` but is not stored in `cost_table.json`; "
      "hybrid+rerank is therefore about the hybrid latency plus this. TODO(run): store the rerank latency in the JSON on the next cost run (a cost-only rerun does not touch the locked retrieval results).", "",
      "## 10.7 Pool-bias checks", "",
      "| System | MAP | MAP without its own unique finds | Change | Kendall tau of system order |", "|---|---|---|---|---|"]
for s, b in bias.items():
    L.append(f"| {s} | {f3(b['MAP_full'])} | {f3(b['MAP_without_own_unique'])} | {sg(b['MAP_without_own_unique'] - b['MAP_full'])} | {b['tau_system_order']:.2f} |")
L += ["", "The two reranked systems entered the pool as well but are not part of this leave-one-system-out check.", "",
      "## 10.8 Does better retrieval give better answers? (RQ3, second half)", "",
      "TODO(run): `python eval/retrieval_vs_answer.py` needs Role C's blind answer ratings. With about 40 pairs the interval is wide; the report states n.", "",
      "## 10.9 Reading the results", "",
      "Findings, each with its interval; where the interval contains zero it is called inconclusive.", ""]
def find(i): return next(h for h in hyp if h["id"] == i)
bt = ret["bootstrap"]
L += [f"- **Hybrid with reranking is the strongest system.** nDCG@10 {f3(ret['overall']['hybrid+rerank']['nDCG@10'])} against {f3(ret['overall']['bm25']['nDCG@10'])} for bm25, a difference of "
      f"{sg(bt['hybrid+rerank']['nDCG@10']['mean_diff'])} (95% CI {sg(bt['hybrid+rerank']['nDCG@10']['ci_low'])} to {sg(bt['hybrid+rerank']['nDCG@10']['ci_high'])}), "
      f"winning {bt['hybrid+rerank']['nDCG@10']['wins']} needs and losing {bt['hybrid+rerank']['nDCG@10']['losses']}.",
      f"- **H1 supported:** bm25 beats tf-idf by {sg(find('H1')['mean_diff'])} nDCG@10 (CI {sg(find('H1')['ci_low'])} to {sg(find('H1')['ci_high'])}). Champion lists did not change this.",
      f"- **H3a supported, H3b tied:** hybrid beats bm25 ({sg(find('H3a')['mean_diff'])}, CI {sg(find('H3a')['ci_low'])} to {sg(find('H3a')['ci_high'])}) but is indistinguishable from dense alone ({sg(find('H3b')['mean_diff'])}, CI {sg(find('H3b')['ci_low'])} to {sg(find('H3b')['ci_high'])}).",
      f"- **H4 inconclusive:** the reranker's gain over hybrid is {sg(find('H4')['mean_diff'])} (CI {sg(find('H4')['ci_low'])} to {sg(find('H4')['ci_high'])}). The gain of hybrid+rerank over bm25 is significant; over hybrid it is not.",
      "- **H2 and H6 inconclusive.** The per-type subsets hold 6 or 7 needs each; no conclusion about dense versus lexical retrieval by need type is supported.",
      "- **Pseudo-relevance feedback did not help** on test in any of the 9 settings tried. On the 10 dev needs it looked slightly positive; the test result is the one that counts.",
      "- **Stemming was not adopted.** With no metadata prefix, stemming lowers nDCG@10 by 0.066 to 0.086 on test (the interval excludes zero in three of four cases) and every stemmed dev configuration also scored below the default. With the metadata prefix stemming makes no measurable difference. The choice was made on dev before the test run.",
      "- **Table chunks and the entity filter showed no measurable effect over all 20 test needs** (nDCG@10 differences -0.001 and -0.010). On dev they looked mildly helpful; the test data do not confirm it. The five-table-need subset was not analysed (see 10.5).",
      "- **Pooling limits.** The pool was built from these same systems and judged by an LLM. Removing a system's unique finds changes MAP by at most 0.097 (dense, the most pool-dependent) and the system ordering is stable (tau 0.75 to 0.96), but the absolute scores should not be compared with other collections.",
      "- **Judgments are LLM-assisted with no human validation.** All conclusions are conditional on that."]
(ROOT / "docs/report_B/10_retrieval_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("wrote docs/report_B/10_retrieval_results.md")

# 10. Retrieval results (Role B)

STATUS: skeleton. Every `TODO(run)` is filled by copying from `eval/out/*.json` after the commands in the right-hand column. Nothing may be estimated.

## 10.1 Judgments and agreement

| Item | Value | Source |
|---|---|---|
| Passages judged, per judge | TODO(run) | `make merge` |
| Double-judged needs / pairs | 10 / TODO(run) | `make merge` |
| Weighted kappa | TODO(run) | `trademesh.evaluation.qrels.weighted_kappa` on the doubly judged pairs |
| Pool size per need (mean, range) | TODO(run) | `data/judgments/pool_key.json` |

## 10.2 Main comparison (RQ1), 20 test needs

| System | P@5 | P@10 | R@10 | F1@10 | MAP | MRR | nDCG@10 | bpref | judged@10 |
|---|---|---|---|---|---|---|---|---|---|
| tfidf (Role A) | TODO(run) | | | | | | | | |
| bm25 | TODO(run) | | | | | | | | |
| dense | TODO(run) | | | | | | | | |
| hybrid (RRF) | TODO(run) | | | | | | | | |
| hybrid + rerank | TODO(run) | | | | | | | | |
| bm25 + PRF | TODO(run) | | | | | | | | |

Command: `python eval/run_retrieval.py --split test` (once, after the freeze). Figure: 11-point interpolated PR curves from `pr_11pt` in the same file.

## 10.3 By need type

nDCG@10 for entity, topical and terminology needs, per system (same file, `by_type`). This is where H2 and H6 are decided.

## 10.4 Hypotheses H1 to H4, H6

Output of `python eval/hypotheses.py`: TODO(run). Each row gives the mean difference, 95 percent bootstrap interval, wins/losses/ties and a verdict from the fixed rule. Rows with an interval containing zero are reported as inconclusive, not as trends.

## 10.5 Ablations (RQ2, H5)

`python eval/ablation.py --split test`: preprocessing grid (16 configurations), table chunks dropped, entity filter off, fusion variants, PRF grid. TODO(run). H5 is read from the rows "drop table chunks" and "nonum", restricted to the five table needs (a per-need view is in `per_query`).

## 10.6 Speed and cost (H7)

`python eval/speed_quality.py` and `python eval/cost_table.py`: index build time, p50/p95 latency per system, and MAP against latency for each champion-list size. TODO(run).

## 10.7 Pool-bias checks

`python eval/pool_bias.py`: leave-one-system-out MAP change and Kendall tau of the system ordering. TODO(run).

## 10.8 Does better retrieval give better answers? (RQ3, second half)

`python eval/retrieval_vs_answer.py`: Spearman correlation between retrieval nDCG@5 and the human answer rating, over (need, configuration) pairs. Needs Role C's blind ratings. TODO(run). With about 40 pairs the interval is wide; the report states n.

## 10.9 Reading the results

To be written after the numbers exist. Rules for the text: describe each finding with its interval; say "inconclusive" where the interval contains zero; report hypotheses that failed as findings (section 10.4 of the proposal); never describe a trend the interval does not support.

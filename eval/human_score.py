"""Score the HUMAN ratings and compare them with the earlier model-assisted ones.

    python eval/human_score.py                 # all raters found under eval/human/raters/
    python eval/human_score.py --raters Shashwat,Ruchir

Reads eval/human/blind_answers.json + key_DO_NOT_OPEN.json (the key only to join labels to systems), the raters' CSVs,
eval/out/rag_runs_test.json (for the automatic number-grounding and verifier results) and, for the comparison only, the
model-assisted sheets eval/out/rating_sheet.csv + rating_key.json.

Writes eval/out/rag_scores_human.json and docs/report_C/human_rating_results.md (ready to paste into the report).
It never fills in a rating: a missing rating is reported as missing and the headline refuses to run on incomplete sheets
unless you pass --allow-partial.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import OUT, ROOT, read_csv, write_json  # noqa: E402
from eval.rag_score import cohen_kappa  # noqa: E402

HUMAN = ROOT / "eval" / "human"
CONFIGS = ["closed_book", "bm25_rag", "best_rag"]
NAMES = {"closed_book": "Closed-book", "bm25_rag": "BM25-RAG", "best_rag": "Best-RAG (hybrid + rerank)"}
PAIRS = [("bm25_rag", "closed_book"), ("best_rag", "closed_book"), ("best_rag", "bm25_rag")]
JUDG = ROOT / "data" / "judgments"


def _rows(path: Path) -> list[dict]:
    return read_csv(path) if path.exists() else []


def load_rater(rdir: Path, key: dict) -> dict:
    ratings = {(r["question_id"], key[r["question_id"]][r["label"]]): int(r["rating"])
               for r in _rows(rdir / "ratings.csv") if r["rating"].strip() != ""}
    cites = {(r["question_id"], key[r["question_id"]][r["label"]], r["citation"]): int(r["supports"])
             for r in _rows(rdir / "citations.csv") if r["supports"].strip() != ""}
    return {"name": rdir.name, "ratings": ratings, "cites": cites}


def paired_bootstrap(per: dict, qids: list[str], resamples: int = 10000, seed: int = 358) -> dict:
    rng = random.Random(seed)
    out = {}
    for a, b in PAIRS:
        d = [per[a][q] - per[b][q] for q in qids]
        means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(resamples))
        out[f"{a} - {b}"] = {"mean_diff": sum(d) / len(d), "ci95": [means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]],
                             "better": sum(x > 0 for x in d), "worse": sum(x < 0 for x in d), "tied": sum(x == 0 for x in d)}
    return out


def leave_one_out(per: dict, qids: list[str]) -> dict:
    """Range of each mean difference when one question is dropped at a time (no question is special-cased)."""
    out = {}
    for a, b in PAIRS:
        diffs = [sum(per[a][q] - per[b][q] for q in qids if q != drop) / (len(qids) - 1) for drop in qids]
        out[f"{a} - {b}"] = {"min": min(diffs), "max": max(diffs)}
    return out


def rq3_with_human(pairs: list, order: list[tuple[str, str]], human: dict) -> dict:
    """RQ3 (does better retrieval give better answers?) recomputed with the human ratings.

    `pairs` is eval/out/retrieval_vs_answer.json: [nDCG@5, answer rating] per (need, configuration), in `order`. Only the
    nDCG@5 values are reused; the answer ratings are replaced by the human ones. Interval: Fisher z approximation."""
    import math

    from trademesh.evaluation.metrics import spearman
    xs = [p[0] for p in pairs]
    ys = [human[k] for k in order]
    rho = spearman(xs, ys)
    z, se = math.atanh(max(-0.999999, min(0.999999, rho))), 1.06 / math.sqrt(len(xs) - 3)
    return {"spearman": rho, "n": len(xs), "ci95_approx": [math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)]}


def retrieval_validation(names: list[str]) -> dict | None:
    """Agreement between each human's relevance grades and the model-assisted grades, on the passages the human judged."""
    llm = {}
    for judge in ("AI", "AI2"):
        f = JUDG / f"pool_sheet_{judge}.csv"
        if f.exists():
            llm[judge] = {(r["query_id"], r["passage_id"]): r["rel"] for r in read_csv(f) if r["rel"].strip() != ""}
    out = {}
    for name in names:
        f = HUMAN / "judgments" / f"pool_sheet_{name}.csv"
        if not f.exists():
            continue
        human = {(r["query_id"], r["passage_id"]): r["rel"].strip() for r in read_csv(f) if r["rel"].strip() != ""}
        if not human:
            continue
        for judge, grades in llm.items():
            common = sorted(set(human) & set(grades))
            if not common:
                continue
            h = [int(human[k]) for k in common]
            m = [int(grades[k]) for k in common]
            conf = {f"human {a} / model {b}": sum(1 for x, y in zip(h, m) if x == a and y == b) for a in (0, 1, 2) for b in (0, 1, 2)}
            out[f"{name} vs {judge}"] = {"n": len(common), "identical": sum(x == y for x, y in zip(h, m)) / len(common),
                                          "kappa": cohen_kappa(h, m), "mean_human": sum(h) / len(h), "mean_model": sum(m) / len(m),
                                          "same_relevant_or_not": sum((x > 0) == (y > 0) for x, y in zip(h, m)) / len(common),
                                          "confusion": conf}
    return out or None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raters", default="", help="comma-separated names (default: every folder under eval/human/raters)")
    ap.add_argument("--allow-partial", action="store_true", help="score incomplete sheets (for a progress check only)")
    ap.add_argument("--no-md", action="store_true", help="do not write docs/report_C/human_rating_results.md")
    args = ap.parse_args()

    blind = json.loads((HUMAN / "blind_answers.json").read_text(encoding="utf-8"))
    key = json.loads((HUMAN / "key_DO_NOT_OPEN.json").read_text(encoding="utf-8"))
    runs = json.loads((OUT / "rag_runs_test.json").read_text(encoding="utf-8"))
    qids = [q["id"] for q in blind["questions"]]
    qtype = {q["id"]: q["type"] for q in blind["questions"]}
    names = [n.strip() for n in args.raters.split(",") if n.strip()] or sorted(p.name for p in (HUMAN / "raters").glob("*") if p.is_dir())
    raters = [load_rater(HUMAN / "raters" / n, key) for n in names]
    raters = [r for r in raters if r["ratings"]]
    if not raters:
        sys.exit("TODO(run): no human ratings yet. Rate the answers with `streamlit run eval/human_app.py` first.")
    want = len(qids) * len(CONFIGS)
    for r in raters:
        print(f"{r['name']}: {len(r['ratings'])} of {want} answers rated, {len(r['cites'])} citations checked")
    complete = [r for r in raters if len(r["ratings"]) == want]
    if len(complete) < len(raters) and not args.allow_partial:
        sys.exit("Some sheets are incomplete. Finish them, or pass --allow-partial for a progress check (never report that).")
    raters = complete or raters

    # answer ratings: per-question mean over raters
    raw = defaultdict(lambda: defaultdict(list))
    for r in raters:
        for (q, cfg), v in r["ratings"].items():
            raw[cfg][q].append(v)
    per = {c: {q: sum(v) / len(v) for q, v in raw[c].items()} for c in CONFIGS}
    common_q = [q for q in qids if all(q in per[c] for c in CONFIGS)]
    configs = {}
    for c in CONFIGS:
        vals = [per[c][q] for q in common_q]
        # citation precision: a citation counts as supporting only if every rater said so
        cite_rows = defaultdict(list)
        for r in raters:
            for (q, cfg, ref), v in r["cites"].items():
                if cfg == c:
                    cite_rows[(q, ref)].append(v)
        full = [v for v in cite_rows.values() if len(v) == len(raters)]
        with_nums = [run[c] for run in runs["runs"] if run[c].get("n_numbers")]
        passed = [run[c]["verifier_passed"] for run in runs["runs"] if run[c].get("verifier_passed") is not None]
        configs[c] = {"n_rated": len(vals), "mean_rating": sum(vals) / len(vals), "share_rated_2": sum(v == 2 for v in vals) / len(vals),
                      "citation_precision": (sum(all(x == 1 for x in v) for v in full) / len(full)) if full else None,
                      "n_citations_judged": len(full),
                      "number_grounding": (sum(1 for x in with_nums if x["numbers_grounded"]) / len(with_nums)) if with_nums else None,
                      "verifier_pass_rate": (sum(passed) / len(passed)) if passed else None}
    report: dict = {"split": "test", "rated_by": [r["name"] for r in raters], "configs": configs, "n_questions": len(common_q),
                    "bootstrap": paired_bootstrap(per, common_q) if len(common_q) == len(qids) else None}

    if report.get("bootstrap"):
        report["leave_one_out"] = leave_one_out(per, common_q)

    # RQ3 with human ratings: reuse the stored nDCG@5 values, but only if they line up with the model-rated run exactly
    rq3_path, llm_path0 = OUT / "retrieval_vs_answer.json", OUT / "rating_sheet.csv"
    if rq3_path.exists() and llm_path0.exists() and (OUT / "rating_key.json").exists() and len(raters) == 1:
        from eval.common import NEEDS
        pairs = json.loads(rq3_path.read_text(encoding="utf-8"))["pairs"]
        test_ids = [n["id"] for n in read_csv(NEEDS / "needs.csv") if n["split"] == "test"]
        order = [(q, c) for c in ("bm25_rag", "best_rag") for q in test_ids]
        lk = json.loads((OUT / "rating_key.json").read_text(encoding="utf-8"))
        lr = defaultdict(list)
        for row in read_csv(llm_path0):
            if row["rating"].strip() != "":
                lr[(row["question_id"], lk[row["question_id"]][row["label"]])].append(int(row["rating"]))
        aligned = len(order) == len(pairs) and all(k in lr and abs(sum(lr[k]) / len(lr[k]) - p[1]) < 1e-9 for k, p in zip(order, pairs))
        if aligned and all(k in raters[0]["ratings"] for k in order):
            report["rq3_human"] = rq3_with_human(pairs, order, raters[0]["ratings"])

    # agreement between humans
    if len(raters) >= 2:
        a, b = raters[0]["ratings"], raters[1]["ratings"]
        ks = sorted(set(a) & set(b))
        report["kappa_between_humans"] = cohen_kappa([a[k] for k in ks], [b[k] for k in ks])
        report["n_double_rated"] = len(ks)

    # comparison with the model-assisted ratings (never used to compute the human scores above)
    llm_path = OUT / "rating_sheet.csv"
    if llm_path.exists() and (OUT / "rating_key.json").exists():
        lkey = json.loads((OUT / "rating_key.json").read_text(encoding="utf-8"))
        llm = defaultdict(dict)
        for row in read_csv(llm_path):
            if row["rating"].strip() != "":
                llm[row["rater"]][(row["question_id"], lkey[row["question_id"]][row["label"]])] = int(row["rating"])
        report["vs_model"] = {}
        for judge, lr in llm.items():
            for r in raters:
                ks = sorted(set(lr) & set(r["ratings"]))
                if ks:
                    h, m = [r["ratings"][k] for k in ks], [lr[k] for k in ks]
                    report["vs_model"][f"{r['name']} vs model pass {judge}"] = {
                        "n": len(ks), "identical": sum(x == y for x, y in zip(h, m)) / len(ks), "kappa": cohen_kappa(h, m),
                        "mean_human": sum(h) / len(h), "mean_model": sum(m) / len(m),
                        "human_higher": sum(x > y for x, y in zip(h, m)), "human_lower": sum(x < y for x, y in zip(h, m))}
    report["retrieval_validation"] = retrieval_validation([r["name"] for r in raters] or names)
    path = write_json("rag_scores_human.json", report)
    print(f"wrote {path.relative_to(ROOT)}")
    for c in CONFIGS:
        e = configs[c]
        print(f"  {NAMES[c]:28s} mean {e['mean_rating']:.3f}  rated 2: {e['share_rated_2']:.0%}  citation precision "
              f"{'n/a' if e['citation_precision'] is None else format(e['citation_precision'], '.3f')} ({e['n_citations_judged']} judged)")
    if report.get("bootstrap"):
        for k, v in report["bootstrap"].items():
            print(f"  {k}: {v['mean_diff']:+.3f}  95% CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  better {v['better']} / worse {v['worse']} / tied {v['tied']}")
    if report.get("rq3_human"):
        r3 = report["rq3_human"]
        print(f"  RQ3 with human ratings: Spearman {r3['spearman']:+.3f} over {r3['n']} pairs, approx 95% CI [{r3['ci95_approx'][0]:+.2f}, {r3['ci95_approx'][1]:+.2f}]")
    for k, v in (report.get("leave_one_out") or {}).items():
        print(f"  leave-one-question-out {k}: {v['min']:+.3f} to {v['max']:+.3f}")
    for k, v in (report.get("vs_model") or {}).items():
        print(f"  {k}: identical {v['identical']:.0%}, kappa {v['kappa']:.3f}")
    if not args.no_md:
        write_markdown(report, qtype, common_q, per)


def write_markdown(rep: dict, qtype: dict, qids: list[str], per: dict) -> None:
    f = lambda x: "n/a" if x is None else f"{x:.3f}"
    who = ", ".join(rep["rated_by"])
    L = ["# RAG study: human ratings", "",
         f"Rated by {who}, each blind to which system wrote an answer (the order of the three answers was shuffled per question; "
         "closed-book answers can still be recognised because they carry no citations). Reference answers were written by the rater "
         "from the filings before rating. Generated by `eval/human_score.py` from `eval/out/rag_scores_human.json`; do not edit the numbers by hand.", "",
         f"## Answer quality ({rep['n_questions']} test questions)", "",
         "| Configuration | Mean rating (0-2) | Rated 2 | Citation precision | Number grounding |", "|---|---|---|---|---|"]
    for c in CONFIGS:
        e = rep["configs"][c]
        L.append(f"| {NAMES[c]} | {e['mean_rating']:.3f} | {e['share_rated_2']:.0%} | {f(e['citation_precision'])} ({e['n_citations_judged']} judged) | {f(e['number_grounding'])} |")
    if rep.get("bootstrap"):
        L += ["", "Paired bootstrap over questions (10,000 resamples, difference in mean rating, 95% interval):", "",
              "| Comparison | Difference | 95% interval | Better / worse / tied |", "|---|---|---|---|"]
        for k, v in rep["bootstrap"].items():
            a, b = k.split(" - ")
            L.append(f"| {NAMES[a]} minus {NAMES[b]} | {v['mean_diff']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | {v['better']} / {v['worse']} / {v['tied']} |")
    if "kappa_between_humans" in rep:
        L += ["", f"Cohen's kappa between the two human raters: {rep['kappa_between_humans']:.3f} on {rep['n_double_rated']} answers."]
    if rep.get("leave_one_out"):
        L += ["", "Leave-one-question-out range of each mean difference (a robustness check; no question is treated specially):", ""]
        for k, v in rep["leave_one_out"].items():
            a, b = k.split(" - ")
            L.append(f"* {NAMES[a]} minus {NAMES[b]}: {v['min']:+.3f} to {v['max']:+.3f}")
    if rep.get("rq3_human"):
        r3 = rep["rq3_human"]
        L += ["", f"**Does better retrieval give better answers (RQ3), with human ratings.** Spearman correlation between retrieval quality (nDCG@5 of the "
                  f"retriever behind each RAG configuration, from the model-assisted relevance judgments) and the human answer rating: "
                  f"{r3['spearman']:+.3f} over {r3['n']} (question, configuration) pairs, approximate 95% interval "
                  f"[{r3['ci95_approx'][0]:+.2f}, {r3['ci95_approx'][1]:+.2f}] (Fisher z). No relation can be claimed."]
    if rep.get("vs_model"):
        L += ["", "## Comparison with the earlier model-assisted ratings", "",
              "| Pair | Answers | Identical | Cohen's kappa | Mean (human / model) | Human higher / lower |", "|---|---|---|---|---|---|"]
        for k, v in rep["vs_model"].items():
            L.append(f"| {k} | {v['n']} | {v['identical']:.0%} | {v['kappa']:.3f} | {v['mean_human']:.2f} / {v['mean_model']:.2f} | {v['human_higher']} / {v['human_lower']} |")
    if rep.get("retrieval_validation"):
        L += ["", "## Validation of the model-assisted retrieval judgments (sample)", "",
              "| Pair | Passages | Identical grade | Same relevant-or-not | Cohen's kappa | Mean grade (human / model) |", "|---|---|---|---|---|---|"]
        for k, v in rep["retrieval_validation"].items():
            L.append(f"| {k} | {v['n']} | {v['identical']:.0%} | {v['same_relevant_or_not']:.0%} | {v['kappa']:.3f} | {v['mean_human']:.2f} / {v['mean_model']:.2f} |")
    L += ["", "## Citation precision and number grounding", "",
          "Citation precision is the share of cited passages that the rater judged to support the sentence they are cited for "
          "(with several raters, a citation counts only if all agree). Number grounding is automatic (the verifier) and does not depend on the rater."]
    (ROOT / "docs" / "report_C" / "human_rating_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote docs/report_C/human_rating_results.md")


if __name__ == "__main__":
    main()

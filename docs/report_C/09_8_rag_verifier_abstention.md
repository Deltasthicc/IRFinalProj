# 9.8 RAG, verifier and abstention

Retrieval-augmented generation lets a language model condition on retrieved text [11]. A fluent model can still invent a figure or a citation, and question answering over SEC filings is hard even with retrieval [13]. We therefore wrap generation in a deterministic check and give the system a way to say that it does not know.

## Context and prompt

The context is the top five passages, labelled `[P1]` to `[P5]`, each with a header of the form `TICKER | FY2025 | Item 1A | text`. In compare mode (two or more companies named) each company contributes its best three passages, grouped by company, so up to six passages are shown. The system prompt tells the model to use only the passages, to put the label of the supporting passage after every claim, to copy numbers exactly as written, to answer in one to four sentences, and, if the passages are not enough, to output exactly `INSUFFICIENT EVIDENCE`. The model runs at temperature 0 and every call is cached on disk by the hash of (model, prompt, parameters), so a run is reproducible and the demo works offline. The **closed-book baseline** uses the same model and question with no passages and no citations.

## Verifier

The verifier uses no language model. For an answer `a` and context `P1..Pn` it checks:

1. **Citations exist.** Every `[Pi]` in `a` has 1 ≤ i ≤ n.
2. **Citations are present.** `a` cites at least one passage; sentences with no citation are listed.
3. **Numbers are grounded.** Every number in `a` occurs in a cited passage.

Numbers are normalised before comparison: `$` and thousands separators are removed, `%` and "percent" are ignored, and the words *thousand, million, billion, trillion* (and `bn`, `mn`) scale the value, so `$1.2 billion` and `1,200 million` are the same number. If a passage says "(in thousands)", "(in millions)" or "(in billions)", its plain numbers also count at that scale. The fiscal year in the passage header counts as present. Item numbers, "10-K" and the labels themselves are not treated as claims. Matching is exact on the normalised value, because the prompt asks for verbatim figures; a rounded figure ("$383.3 billion" for 383,285 million) is therefore flagged. This is strict by design and is listed under limitations.

A failed check is **flagged in the trust panel and never hidden or repaired**. For each number the panel shows whether it was found in a cited passage, found only in an uncited passage, or not found.

## Abstention

The system abstains if the model outputs `INSUFFICIENT EVIDENCE`, or if the best score is below a threshold τ:

abstain ⇔ conf(q) < τ  or  answer = INSUFFICIENT EVIDENCE.

Until the cross-encoder is available the score is conf(q) = s₁ / |T(q)|, where s₁ is the best first-stage retrieval score and |T(q)| the number of distinct content words in the query (so long queries are not favoured); when the reranker is installed its best score is used instead and the rule is unchanged. τ is tuned on the development queries only: candidates are 0 and the midpoints between consecutive observed confidences, we maximise F1 with the positive class "should abstain", and ties go to the smaller τ. A score below τ skips the model call, which also saves cost.

## Evaluation protocol

The test needs (20) are run once, after the freeze, in three configurations: closed-book, BM25 context and the best retriever with the reranker. Two of us rate each answer 0, 1 or 2 against the reference, blind: the order of the three answers is shuffled separately for every question and the system names are hidden (presentation bias, as in the reputation-systems lecture). We report mean rating, the share rated 2, Cohen's κ on the double-rated answers, **citation precision** (share of cited passages that a human judges to support the claim) and **number grounding** (share of answers with numbers whose numbers are all found in cited passages, computed automatically). For abstention we use 20 answerable and 10 unanswerable questions with the positive class "should abstain" and report precision, recall, F1 and accuracy; τ is fixed on the dev items, so the headline numbers are on the test items. Because the test collection is small we report per-question outcomes next to means and make no significance claims without support.

Everything is implemented in `src/trademesh/rag/` and `eval/rag_*.py`, `eval/abstention_eval.py`.

# Audit trail for the RAG answer ratings and citation checks (test split)

**Everything in this folder is LLM-assisted. No human rated these answers or checked these citations.**
Both passes were made by the same language model (Claude), so agreement between them is a consistency
check, not independent validation.

## Files
- `rag_rating_audit.csv`   one row per (question, blind label A/B/C): rating by pass 1 (`AI`) and pass 2 (`AI2`).
- `rag_citation_audit.csv` one row per (answer, cited passage): `supports` by each pass and the consolidated value (the lower of the two).
The labels A/B/C are the anonymous labels of `rating_sheet.csv`; the mapping to systems is `eval/out/rating_key.json`,
which the rater did not open before or during rating.

## Scale and rules
- Answer rating (0/1/2) as in `eval/rag_sheets.py`: 2 = correct and complete, 1 = partly correct or incomplete,
  0 = wrong, invented, or no answer. An abstention on an answerable question is 0.
- Reference evidence: the passages graded 2 and 1 for that question in `data/judgments/qrels.csv` (itself LLM-assisted),
  plus the text of the cited passages. A claim that states a specific (a named company, a number, a mechanism)
  that the corpus evidence does not contain is treated as unsupported and cannot earn a 2.
- Citation `supports` = 1 if the cited passage states the main assertion of the sentence it is cited for. When one sentence
  is cited jointly to several passages, each passage counts if it supports its part.

## Passes
- Pass 1 (`AI`): questions in sheet order; evidence shown as above.
- Pass 2 (`AI2`): questions in a shuffled order, passages shown without truncation, and a stricter checklist for a 2
  (answers the question asked, no claim unsupported by the corpus, covers the principal points of the top relevant passages).
- The scorer (`eval/rag_score.py`) averages the two raters per answer. Citation precision uses the consolidated (lower) value.

## Agreement
- Answers: 56 of 60 identical. Cohen's kappa 0.886 (unweighted, as in `rag_score.py`); 0.906 linear-weighted; 0.930 quadratic-weighted.
  The four disagreements are all 2 versus 1: E06-C, E09-B, T09-C, R04-A.
- Citations: 94 of 95 identical. The one difference (E08, label B, cited passage P4) is a passage that does not contain
  the "rate will accelerate" part of a claim cited to four passages; the consolidated value is 0.

## Known limits (state these in the report)
- Same model for both passes, and pass 2 was not blind to the existence of pass 1, so agreement is probably optimistic.
- Not fully blind to the system: closed-book answers can usually be recognised because they have no citations.
- The rater and the generator are different model families (Claude rated, Gemini generated), which reduces self-preference,
  but an LLM rater can still favour fluent, grounded-looking text.
- In pass 1 the on-screen text of a few long passages was cut at roughly 1,000-1,100 characters (questions T09, T10, R05);
  the cut-off tails were re-read in pass 2 and no rating changed.
- 20 test questions only (about 6-7 per question type), so per-type numbers are descriptive.

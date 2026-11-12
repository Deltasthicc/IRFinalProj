# 9.6 Query processing, entities, section prior, table chunks

## Company entities with spelling tolerance

Analysts misspell company names ("Telsa", "Micorsoft"), and a misspelt name silently removes the best filter we have. We therefore detect company mentions with an edit-distance matcher built from scratch (no fuzzy-matching library), following the dictionary and edit-distance material of Manning et al. [17].

*Dictionary.* Each company has a name, a ticker and aliases. Every string is reduced to a key: lower-case letters and digits only ("Coca-Cola" becomes `cocacola`). The legal suffix is also dropped ("Apple Inc." gives `apple`).

*Bigram shortlist.* Every key is padded as `$key$` and split into character bigrams, and an index maps each bigram to the entries containing it. For an input key `q` we collect the entries that share at least one bigram and rank them by Jaccard overlap

J(q, e) = |B(q) ∩ B(e)| / |B(q) ∪ B(e)|,

where B(·) is the set of bigrams. Entries with J < 0.2 are dropped and the best ten are kept.

*Levenshtein confirmation.* The shortlist is confirmed with the unit-cost edit distance computed by the standard two-row dynamic program. A candidate is accepted if the distance is at most 2, and at most 1 when either string has four characters or fewer (this covers tickers, where two edits would map almost any string onto some other ticker). Ties are broken by distance, then by Jaccard overlap; if two different companies tie on both, we refuse to guess.

*Finding mentions in a question.* We slide a window of one to four words over the question, longest first, and accept the first window that matches. Three guards limit false positives: a word that already occurs in the corpus vocabulary is never "corrected" (so "apply" does not become "Apple"); a window may not start or end with a function word; and tickers of three letters or fewer count only when typed in capitals ("KO", "BA").

*Use.* The detected companies are removed from the question to give the *topic query*, and retrieval is restricted to those companies (entity filter). The interface tells the user when a correction was made ("Did you mean Tesla?").

*Evaluation.* On 40 typo'd company names and tickers the matcher reaches **39/40 = 0.975** correct (0 wrong company, 1 no match); the bigram shortlist agrees with a brute-force Levenshtein scan over the whole dictionary on all 40 inputs (`eval/entity_eval.py`, output in `eval/out/entity_eval.json`). The one failure is the ticker typo `MSTF` for `MSFT`: a transposition costs two edits in plain Levenshtein distance, which the one-edit limit for short strings rejects. A Damerau variant that counts a transposition as one edit would fix it; we kept plain Levenshtein, as planned, and report the failure. The typo set was written by us, so this measures the matcher on typo types we thought of, not on real user typos. The result is the same with the proposal company list and with the final `data/companies.json` (39/40 both times), and the 40 inputs were not changed between the two runs.

## Section prior

Filings differ in how often a section holds what analysts ask about: risk factors (Item 1A) and management's discussion (Item 7) answer many questions, while mine-safety disclosures or exhibit lists rarely do. This is a static quality score in the sense of Manning et al. [17]. We add it to the retrieval score:

final(q, d) = base(q, d) + λ · g(section(d)),  g ∈ [0, 1].

`base` is min-max normalised over the candidate list first, so λ means the same for BM25, dense and hybrid scores. `g` is set by hand per Item (for example 1.0 for Items 1A and 7, 0.8 for Items 1 and 8, 0.1 for Items 1B, 4 and 9). `tune_lambda` in `query/section_prior.py` can choose λ from {0, 0.05, 0.1, 0.2, 0.3, 0.5} by mean nDCG@10 on the 10 development queries, with ties going to the smaller λ. We implemented and unit-tested the prior but did not run that tuning or evaluate its effect: λ = 0 in every reported run, so the section prior is not part of any result in this report and we make no claim about it.

## Table chunks, as they relate to querying

Each table is one passage (caption, header row and rows as text), so a figure stays attached to its row and column labels. This matters for querying in two ways. First, a question such as "What was Microsoft's revenue by segment?" can be answered from a single retrieved chunk. Second, the verifier knows the scale of a table: if a chunk says "(in millions)", its plain numbers also count at that scale, so "383,285" in the table supports "383,285 million" in an answer (section 9.8).

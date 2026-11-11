# Audit of the LLM-assisted relevance judgments

All 589 pooled (need, passage) pairs were graded 0/1/2 by an LLM (Claude), blind to which system retrieved
what (`data/judgments/pool_sheet_AI.csv`, judge "AI"). A second, independent LLM pass graded a stratified
91-pair subset (`pool_sheet_AI2.csv`, judge "AI2"). No human graded any pair.

`ai_pass_agreement.csv` holds the 91 pairs with four columns:
- `*_initial`: the grades as first assigned, before any reconciliation.
- `*_corrected`: grades after a truncation error was found. Both passes first viewed only the opening
  characters of long passages, so a few relevant passages were graded 0. Pass 1 was corrected for 25
  passages after reading every zero-graded long passage in full; pass 2 for 1 passage.

Report the INITIAL agreement as the headline, because the corrected columns are not independent of each other.
Where two judges graded the same pair, `qrels.consolidate` takes the lower grade.

# Human rating guide (the part only a person can do)

The study's answer ratings, citation checks and retrieval judgments were first made by a language model in two passes
(`docs/report_C/audit/`). This folder lets a person redo them **blind**, so the report can say "rated by humans".
Nothing in here contains a model rating, and the tool never shows which system wrote an answer.

## What you will do

| Step | Task | Items | Time (rough) | Required? |
|---|---|---|---|---|
| 1 | Write a short **reference answer** for each test question, from the filings | 20 | 40-60 min | yes |
| 2 | **Rate** the anonymous answers 0 / 1 / 2 against your reference | 60 | 40-60 min | yes |
| 3 | **Check citations**: does the cited passage support the sentence it is cited for? | 95 | 30-45 min | yes |
| 4 | **Validate retrieval**: grade passages 0 / 1 / 2 (a sample of 8 test needs) | 146 | 45-60 min | optional |

If you are short of time: steps 1-3 are the ones that change the headline result. Step 4 only measures how far the model's
relevance judgments agree with yours.

## Rules that keep it honest

* **Stay blind.** Do not open `eval/human/key_DO_NOT_OPEN.json`, `eval/out/rag_runs_test.json`, `eval/out/rating_sheet.csv`,
  `eval/out/rating_key.json` or anything in `docs/report_C/audit/` until you have finished all steps.
  Do not ask the main app the test questions while rating.
* **Judge from the filings**, not from how fluent an answer sounds. The tool gives you a search box over the real filings.
* **Rate every answer on its own**, not relative to the other two.
* Note that answers with no citations are probably closed-book (the tool cannot hide that). Rate them the same way.
* If two people rate (recommended if a teammate has time), each uses their own name and works alone.
  The scorer then reports Cohen's kappa between you.

## The scales

**Reference answer (step 1).** 1 to 2 sentences, in your own words, using only what the filings say. If the filings genuinely do
not answer the question, tick "The filings contain no answer" instead.

**Answer rating (step 2).**
* **2** = correct and complete: answers the question asked, no claim the filings do not support, covers the main points.
* **1** = partly correct, or incomplete: right direction but missing a main point, or one claim you cannot find in the filings.
* **0** = wrong, invented, or no answer. A refusal ("INSUFFICIENT EVIDENCE") is a **0**, unless your reference says the
  filings have no answer, in which case a refusal is correct (2).

**Citation check (step 3).** **Yes** if the cited passage states the main claim of the sentence it is cited for. If one
sentence cites several passages, each passage counts if it supports its own part. **No** if it is only on the same topic.

**Retrieval grade (step 4).** **2** answers the need, **1** partly relevant, **0** not relevant. Judge each passage on its own.

## How to run it

From the project folder (`IRFinalProject`). The project's virtual environment is the `.venv` folder inside it.

```powershell
# PowerShell on Windows (works without activating anything)
.\.venv\Scripts\python.exe -m streamlit run eval/human_app.py

# or activate once, then plain `streamlit` works in that window
.\.venv\Scripts\Activate.ps1
streamlit run eval/human_app.py
# if PowerShell blocks the script ("execution policy"): Set-ExecutionPolicy -Scope Process RemoteSigned
```
```bash
# Git Bash / macOS / Linux, with the environment active
streamlit run eval/human_app.py
```

A browser tab opens. Type your name (letters only) in the sidebar, then work through steps 1, 2 and 3 using the sidebar.
Every click is saved to `eval/human/raters/<your name>/` straight away, so you can close the tab and come back; the sidebar shows
your progress. The sidebar's question list marks finished questions with a tick.

**Step 4 (optional)** uses the existing judging app on your blank sheet, in a second terminal:

```bash
# Git Bash / macOS / Linux
TRADEMESH_JUDGMENTS=eval/human/judgments streamlit run eval/judge_app.py --server.port 8502
```
```powershell
# PowerShell
$env:TRADEMESH_JUDGMENTS = "eval/human/judgments"; .\.venv\Scripts\python.exe -m streamlit run eval/judge_app.py --server.port 8502
```

Pick your name in that app's sidebar. A second person needs their own sheet: `python eval/human_prepare.py --names TheirName --sheets-only`
(this does not touch the answers or labels; never use `--force` once anyone has rated).

## When you are done

```bash
python eval/human_score.py                 # all raters found under eval/human/raters/
python eval/human_score.py --raters Shashwat,Ruchir     # or choose raters
# PowerShell without activating the environment:  .\.venv\Scripts\python.exe eval/human_score.py
```

It refuses to score incomplete sheets (add `--allow-partial` only to check progress). It writes
`eval/out/rag_scores_human.json` and `docs/report_C/human_rating_results.md` (tables ready to paste), and the app's
**Evaluation > RAG answers** tab then shows the human ratings automatically.

Then, by hand, and honestly:

1. Paste the tables from `docs/report_C/human_rating_results.md` into `docs/report_C/10_rag_results.md` under a new heading
   (keep the model-assisted results as the "first study", they are part of the story).
2. In `docs/report_C/11_conclusions_limitations.md` replace the bullet "LLM-assisted ratings, no human validation" with what is now
   true (who rated, how many, kappa between you and the model, whether the ranking of the three systems changed).
3. Search for other claims that say there was no human validation and update them:
   `grep -rn "no human\|LLM-assisted" docs README.md src/trademesh/app`  (the abstract in `docs/report_B/00_abstract_introduction.md`
   and the notice in the app's Evaluation tab are the likely ones).
4. If your ratings disagree with the model's, report **both** and say so. Do not delete or edit the model-assisted files.
5. Commit and push:
   ```bash
   git add eval/human eval/out/rag_scores_human.json docs/report_C
   git commit -m "Add blind human ratings of the RAG study"
   git push
   ```

## Second rater (Ruchir): hand-in steps

1. `git pull` the latest `main` (or the branch you were told to use) so you have the same blind answers as the first rater.
   Never run `eval/human_prepare.py --force`: it would reshuffle the labels and make the existing ratings unusable.
2. Run the rating app (see "How to run it"), type your name exactly as `Ruchir` in the sidebar, and complete steps 1, 2 and 3
   (reference answers, 60 ratings, 95 citation checks). Step 4 is optional; your blank sheet is
   `eval/human/judgments/pool_sheet_Ruchir.csv`.
3. Do not open `eval/human/key_DO_NOT_OPEN.json` or look at the first rater's folder before you have finished.
4. Hand in by pushing your folder to GitHub, on any branch:
   ```bash
   git checkout -b ruchir/human-ratings
   git add eval/human/raters/Ruchir eval/human/judgments/pool_sheet_Ruchir.csv
   git commit -m "Add Ruchir's blind human ratings"
   git push -u origin ruchir/human-ratings
   ```
5. On the first rater's machine, `python eval/watch_ratings.py --once` fetches the remote, checks that every answer, citation
   and reference is filled in and valid, imports the folder, and runs `eval/human_score.py` over both raters (mean rating per
   question, kappa between the two humans, each rater alone). It refuses an incomplete sheet and prints what is missing.
   `python eval/watch_ratings.py --interval 300` keeps checking every five minutes. It imports data only (it never runs
   anything from the remote) and never writes prose: the wording in the report and README is updated by hand from the new numbers.

## Files

| File | What |
|---|---|
| `blind_answers.json` | the 20 questions, three anonymous answers each, the passages each answer cites, hint passages |
| `key_DO_NOT_OPEN.json` | label to system. Open only after finishing; the scorer reads it for you |
| `judgments/pool_sheet_<name>.csv` | your blank retrieval sheet (step 4) |
| `raters/<name>/references.csv, ratings.csv, citations.csv` | your saved work |

The labels were shuffled with a new seed (`python eval/human_prepare.py --seed 1119`), so the order differs from the model-rated
sheets.

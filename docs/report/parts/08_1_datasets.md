## 8.1 Datasets

The project uses four datasets, all built by us from public sources: the filing corpus, the information needs with their relevance judgments, the unanswerable questions, and the misspelt company names. No external training data is used; the dense encoder and the cross-encoder are pre-trained models applied unchanged.

### Filing corpus

**Source.** The most recent Form 10-K of each of 20 large US companies, downloaded from SEC EDGAR (the SEC's public filing database) with a crawler that identifies itself and waits between requests. The companies span technology (Apple, Microsoft, NVIDIA, Intel), consumer and media (Tesla, Amazon, Netflix, Walmart, Coca-Cola, PepsiCo, Disney), banks (JPMorgan, Goldman Sachs, Bank of America), energy (Exxon Mobil, Chevron), health (Pfizer, Johnson & Johnson, UnitedHealth) and aerospace (Boeing). The filing list, with accession numbers and URLs, is in `data/filings.json`.

**Processing.** Each filing's HTML is parsed, split by 10-K Item, and cut into passages; every table is kept whole as one chunk (caption, header row and rows as text). Each passage records its ticker, fiscal year, Item, kind (text or table) and page.

{{CORPUS_TABLE}}

Fiscal years are 2025 for 16 companies and 2026 for four (Apple, Microsoft, NVIDIA and Walmart, whose fiscal years end between January and September 2026), so every question is about the latest filing of each company only. Of the 11,964 passages, 10,912 are text and 1,052 are tables. The distribution over 10-K Items is Item 8 (financial statements) 2,717, Item 7 (management's discussion) 2,409, Item 1A (risk factors) 1,738, Item 15 (exhibits and financial statement schedules) 1,427, Item 1 (business) 1,132, unlabelled or other material 2,014, and the remaining Items (2, 5, 9A, 1C, 7A, 3) together 527. A passage has 122 words on average (median 131, longest 713); the corpus holds 1,461,121 words. The three banks (JPMorgan, Goldman Sachs, Bank of America) contribute 3,756 passages (31%), so a topical question has many similar bank passages to compete with.

### Information needs and relevance judgments

**Needs.** 30 information needs, each a natural-language question and a short keyword query, in three types: entity (about one named company), topical (a theme across companies) and terminology (a defined term or a figure that sits in a table). There are 10 of each type; 10 needs are development needs, used for every tuning choice (3 entity, 3 topical, 4 terminology) and 20 are test needs (7, 7 and 6), scored once after the feature freeze. Five needs, all terminology, require a table. The needs are in `data/needs/needs.csv`.

**Pooling and judgments.** For every need the top 8 results of every system were merged, deduplicated and shuffled; the judging sheet shows the need and the passage and nothing else. The pools hold 11 to 30 passages per need (mean 19.6). Pooled passages were graded 0 (not relevant), 1 (partly) or 2 (answers the need) by a language model working from the sheet alone, and 91 pairs spread over all 30 needs were graded a second time by an independent pass. The relevance file `data/judgments/qrels.csv` has 680 rows (253 of grade 2, 179 of grade 1, 248 of grade 0). **No human graded a relevance pair**; a planned human spot-check was not done. Agreement between the two passes is reported in section 8.2.

**Size.** 30 needs, 20 of them test needs, is small. The cost is wide intervals, which is why every comparison is reported with a bootstrap interval and why several hypotheses are inconclusive.

### Unanswerable questions

10 questions written by us whose answer is not in the corpus (4 development, 6 test): a future fiscal year, personal information not in a 10-K, a prediction, a company outside the corpus, a detail a 10-K does not give (for example a headcount for one country), and a question about an earlier year than the latest filing. They test abstention (section 8.12). They are easy, in the sense that none is a near miss of an answerable question; this is a limitation (section 9).

### Misspelt company names

40 misspelt company names written by us, covering deletion (15), substitution (10), transposition (4), insertion (2), spacing errors (3) and six ticker errors; they test the entity matcher (section 8.12). Because we wrote them, they measure the typo types we thought of, not real user typing.

### Why the corpus was not made larger

More filings or companies would not help answer any of our research questions, and would invalidate results already produced: the test run is locked, the 680 judgments pool from this corpus, the human ratings and the cached demo answers are tied to it. The corpus already exceeds the proposal's target of 6,000 to 10,000 passages. A corpus of several filing years per company would be the natural extension (section 9).

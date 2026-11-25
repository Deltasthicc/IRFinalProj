TITLE: TradeMesh: presentation notes
SUBTITLE: Search and verified question answering over SEC 10-K filings. What it is, how it works, how to run the demo, and how every part of the report connects to the website and the code.

## How to use these notes

These notes are written so that you can explain the project to the instructor in depth, in plain language. Read sections 1 to 3 first (what the project is and how it works), section 4 and 5 for the evaluation and results, and section 6 for the demo. Section 9 has the questions you are most likely to be asked. Every number here is the same as in the report (`docs/report/TradeMesh_Report.pdf`) and comes from a file in `eval/out/`.

Boxes with a coloured edge: <span class="tag say">SAY</span> is a sentence you can say aloud, <span class="tag demo">DEMO</span> is what to click, <span class="tag code">CODE</span> is where it lives in the repository.

## 1. The project at a glance

### 1.1 The problem

A Form 10-K is the yearly report that every listed US company files with the Securities and Exchange Commission. It is long (hundreds of pages), but the answer to a reader's question is usually one sentence or one table row. Two ordinary tools fail the reader:

* **Keyword search** finds passages that contain the words, but it returns text, not answers, and it often mixes up companies (every filing talks about "risk factors" and "revenue").
* **A language model** gives a fluent answer, but it may quote an old figure or invent one, and it does not say where the answer came from.

Filings are the main public source on a company's money and risks, so a wrong or unsupported number can mislead a decision. That is why we care about answers that can be checked.

### 1.2 The idea

> **Search first, language model second, checker last.**

The language model is only allowed to see the five best passages that search found. It must cite them as [P1] to [P5]. A plain program (not a model) then checks the citations and every number. If the evidence is thin, the system says "insufficient evidence" instead of guessing. Because the answer is only as good as the five passages, the quality of **retrieval** decides the quality of the answer, which is why this is an information retrieval project.

### 1.3 What we built

| Part | What it is |
|---|---|
| Search engine from scratch | Positional inverted index with skip pointers, Boolean / phrase / wildcard search, TF-IDF with cosine similarity, champion lists, BM25, spelling-tolerant company matching |
| Modern retrieval | Dense (sentence-vector) search, hybrid search by rank fusion, cross-encoder reranking, pseudo-relevance feedback |
| Answer layer | Cited answers with a language model, a deterministic checker for citations and numbers, a refusal rule, a closed-book baseline, compare-two-companies mode |
| Company graph | Which company's filing mentions which; PageRank and HITS |
| Test collection and evaluation | 30 questions we wrote, pooled relevance judgments, ten retrieval setups compared, pre-declared hypotheses, bootstrap confidence intervals, blind human ratings |
| Application | A six-tab web app (Ask, Search, Graph, Pages, Evaluation, System) and a Docker image |

### 1.4 The numbers to remember

| Fact | Value |
|---|---|
| Companies and passages | 20 companies, 11,964 passages (10,912 text, 1,052 table chunks), 1.46 million words |
| Test questions | 30 needs (10 for tuning, 20 for one locked final test), 10 unanswerable questions, 40 misspelt names |
| Best retriever (hybrid + reranker) | nDCG@10 **0.828**, against 0.699 for BM25 and 0.560 for TF-IDF |
| Answer quality (two humans, 0 to 2) | **1.20** with the best retriever, 0.90 with BM25, **0.575** without retrieval |
| Agreement of the two raters | exact agreement on 41 of 60 answers, kappa 0.39 |
| Spelling correction | 39 of 40 misspelt company names fixed (97.5%) |
| Refusal | declined 6 of 6 unanswerable test questions, 0 of 20 answerable ones |
| Company graph | 20 companies, 14 links, PageRank vs in-degree Spearman 0.93 |

<div class="callout say"><b>SAY:</b> "TradeMesh answers questions over the latest 10-K filings of 20 companies. Every answer cites the passages behind it, a program checks the citations and numbers, and the system says so when the evidence is not enough. The search core is written from scratch from the course material, and we compare it fairly with modern retrieval."</div>

### 1.5 Every IR concept from the course that the project uses

The proposal mapped each lecture to a part of the system. Use this table when the instructor asks "where is the IR?". "Built by us" means we wrote the code from the lecture formulas; "used" means a pre-trained model or an idea from beyond the slides.

| Course topic | Where it appears in TradeMesh | Built by us? |
|---|---|---|
| Boolean retrieval, inverted index, query processing | Positional inverted index; Boolean AND / OR / NOT search (Search tab, retriever `boolean`) | Yes |
| Tokenisation, normalisation, stemming, stop words | Preprocessing ablation grid (stop words, stemming, numbers kept or not, metadata prefix); numbers like 383,285 and 383285 match | Yes |
| Skip pointers, phrase queries | Skip pointers in postings intersection; phrase search through word positions | Yes |
| Wildcard queries | Permuterm index (one star) and 2-gram index (several stars), e.g. `cyber*` | Yes |
| Edit distance, spelling correction | Company-name matching: bigram shortlist (Jaccard) plus Levenshtein; 39 of 40 typos fixed | Yes |
| TF-IDF weighting, vector space model, cosine | Retriever `tfidf` (1 + log10 tf) x log10(N / df) with length-normalised vectors | Yes |
| Heap-based top-K | The K best passages are selected with a heap | Yes |
| Index elimination, champion lists | `tfidf-champ-50` and `tfidf-champ-200`; speed against quality measured | Yes |
| Static quality scores | Section prior: final = base + lambda x g(10-K section) | Yes (not helpful; lambda = 0) |
| Evaluation: precision, recall, F1, MAP, MRR, nDCG, precision-recall curve | All reported for the ten retrievers on 20 test questions | Yes (metrics written and hand-checked) |
| Test collections, pooling, kappa | 30 needs, pools of the top 8 of every system, weighted kappa between two judging passes | Yes |
| Result summaries (snippets) | Highlighted key-word-in-context snippets and "Why this score" in the Search tab | Yes |
| Crawling: politeness, robots, caching | EDGAR crawler with a declared User-Agent, rate limit and disk cache (see the note on robots.txt in the traceability file) | Yes |
| PageRank and HITS (link analysis) | Company mention graph: who mentions whom | Yes |
| Presentation bias in judging (reputation-systems lecture) | Answers and pooled passages were shuffled and system names hidden when judging and rating | Process |
| Beyond the slides: BM25 | Retriever `bm25`, tuned on the tuning questions | Yes |
| Beyond the slides: dense retrieval, cross-encoder | Pre-trained MiniLM encoder and reranker; our search code around them | Used |
| Beyond the slides: rank fusion | Reciprocal rank fusion, CombSUM, CombMNZ | Yes |
| Beyond the slides: Rocchio relevance feedback | Pseudo-relevance feedback (`bm25+prf`) | Yes |
| Beyond the slides: retrieval-augmented generation | Cited answers from five passages, checked by a verifier | Yes (verifier, prompt, cache); model used |
| Beyond the slides: late interaction (ColBERT / ColPali) | MaxSim page scoring for the Pages tab (small demo: page-image search against text search) | Yes (scoring); model used |

## 2. How it works, step by step

@@FIG:docs/report_C/fig_model.svg|165|Figure 1. The pipeline. The top row is built once (offline). The two lower rows run for every question. The dashed box is the page-image track; the report says it was not built, and a small demonstration was added afterwards (section 3.16).@@

### 2.1 Offline stage (done once)

1. **Collect.** A polite crawler downloads the latest 10-K of each of 20 companies from SEC EDGAR. It sends a declared User-Agent, stays under the rate limit and caches every file. The list of filings, with accession numbers and URLs, is in `data/filings.json`.
2. **Parse and chunk.** Each filing is split by its 10-K section (Item 1 business, Item 1A risk factors, Item 7 management discussion, Item 8 financial statements and so on). Text is cut into passages (mean 122 words). **Every table is kept as one chunk** (caption, header row and rows as text) so that a figure stays attached to its row and column labels. Each passage records ticker, fiscal year, Item, kind (text or table) and page.
3. **Index.** From the passages we build the positional inverted index, the TF-IDF and BM25 statistics, the dense sentence vectors (computed once and cached on disk, 18 MB) and the company mention graph.

<div class="callout code"><b>CODE:</b> <code>src/trademesh/ingest/</code> (edgar.py, parse.py, chunk.py, corpus.py); output <code>data/passages.jsonl</code> and <code>data/corpus_report.json</code>.</div>

### 2.2 Online stage (for every question), with a worked example

We follow one real question through the system: **"How does Telsa describe the risk of depending on its Chief Executive Officer?"** (note the typo).

| Step | What happens | Why | In the demo |
|---|---|---|---|
| 1. Understand | The query processor finds the company name "Telsa", corrects it to Tesla, Inc. (ticker TSLA), removes it from the topic words and remembers it as a filter. | Searching for "Telsa" would find nothing; filtering by company stops other companies' risk factors from competing. | A red bar "Corrected Telsa >>> Tesla, Inc."; chip "Entity: TSLA / 1 fixed" |
| 2. Retrieve | BM25 and the dense model each rank the passages; reciprocal rank fusion merges the two lists. Only Tesla passages are kept. | BM25 is strong on exact words and names; the dense model is strong on meaning. Fusion uses both. | Chip "Retrieve: 5 passages" |
| 3. Rerank | A cross-encoder rereads the top 30 candidates together with the question and keeps the best five. | It is more accurate than the first stage but too slow for the whole corpus, so it only sees a short list. | Evidence cards P1 to P5 with scores |
| 4. Answer | The language model gets the five passages labelled [P1] to [P5] and the rule: answer only from these, cite every claim, otherwise reply exactly INSUFFICIENT EVIDENCE. Temperature 0; the call is cached by a hash of (model, prompt, parameters). | The model cannot use facts from outside the filings, and the same question always gives the same answer. | Chip "LLM: Cache" or "Live"; the answer with small P1 tags |
| 5. Check | The verifier tests that every cited [Pn] exists, that the answer is cited, that every sentence is cited, and that every number occurs in a cited passage. | The model's claims become testable by a program that cannot be fooled by fluent text. | The **trust panel** with Pass / Flag per check |
| 6. Decide | If the model said INSUFFICIENT EVIDENCE the system declines; otherwise it shows the answer with the trust panel. | A refusal is better than a confident wrong answer. | Chip "Decision: Answer" (or "Abstain") |

@@FIG:docs/report/figures/fig_answer_flow.png|125|Figure 2. The answer-checking flow. The checker is an ordinary program, not a language model.@@

The answer to the Tesla question is: *"Tesla states that it is highly dependent on the services of Elon Musk, who is the Technoking of Tesla and Chief Executive Officer [P1]. Although Mr. Musk spends significant time with Tesla and is highly active in management, he does not devote his full time and attention to Tesla because he also holds management positions at other companies and is involved in other ventures [P1]."* The checks pass: P1 exists, every sentence is cited, and there are no numbers to verify.

<div class="callout code"><b>CODE:</b> the whole flow is <code>src/trademesh/rag/pipeline.py</code>; prompt and context builder <code>rag/prompt.py</code>; verifier <code>rag/verifier.py</code>; refusal rule <code>rag/abstain.py</code>; language-model client and cache <code>rag/llm.py</code>; company matching <code>query/entities.py</code>.</div>

@@FIGROW:docs/report/figures/app_ask.png|0,0,1140,1270|docs/report/figures/app_search.png|0,0,1140,1010|Figure 3. The web app. Left: the Ask tab (corrected name, cited answer, trust panel, evidence). Right: the Search tab (two retrievers side by side, highlighted words).@@

## 3. The ideas behind each part (how to explain them)

This section explains each technique in plain words, with a small worked example where it helps. The instructor knows the course material, so connect each idea to the lecture it comes from.

### 3.1 Inverted index, positional index and skip pointers (lectures on Boolean retrieval)

An **inverted index** maps each word to the list of passages that contain it (its postings list). To answer "cyber AND risk" you intersect the two lists. A **positional** index also stores the positions of the word inside each passage, which allows **phrase search** ("supply chain" means the two words next to each other: position of "chain" = position of "supply" + 1). **Skip pointers** let the intersection jump over parts of a long list: for a list of length L we place a skip every √L entries, and while merging two sorted lists we follow a skip whenever the landing entry is not larger than the other list's current entry. In our tests, intersecting a long list with a short one used under a fifth of the comparisons.

<div class="callout code"><b>CODE:</b> <code>src/trademesh/index/positional.py</code>. The positional index keeps stop words (removing them would shift positions and break phrases).</div>

### 3.2 Boolean, phrase and wildcard queries

The Boolean retriever accepts AND, OR, NOT (upper case), parentheses, quoted phrases and wildcards. Precedence is NOT, then AND, then OR; two words side by side mean AND. A **wildcard** such as `cyber*` needs the dictionary terms that match: for one star we use a **permuterm index** (every rotation of the word plus a marker, sorted, so the pattern becomes a prefix search); for several stars we use a **2-gram index** that intersects the sets of words containing each 2-gram, and then filter with a regular expression because 2-grams alone allow false matches.

<div class="callout demo"><b>DEMO:</b> Search tab, pick <b>boolean</b>, type <code>"supply chain" AND (semiconductor OR chips)</code> or <code>cyber*</code>.</div>

<div class="callout code"><b>CODE:</b> <code>index/boolean.py</code>, <code>index/wildcard.py</code>.</div>

### 3.3 TF-IDF with cosine similarity (lecture on vector space ranking)

A word should count more if it appears often in a passage (term frequency, tf) and if it is rare in the collection (inverse document frequency, idf). We use

> w(t, d) = (1 + log10 tf) × log10(N / df)

where N = 11,964 passages and df is the number of passages containing t. **Worked example:** a word that occurs 3 times in a passage and in 100 passages overall: (1 + log10 3) × log10(11,964 / 100) = 1.477 × 2.078 = **3.07**. The query is weighted the same way, and a passage is scored by the **cosine** of the angle between the query vector and the passage vector, which removes the effect of passage length. We store passage vectors already length-normalised, so a query needs only dot products, and select the top K with a **heap**.

### 3.4 Champion lists (lecture on index elimination)

For each word we keep only the r passages with the highest weight (its "champions") and, at query time, score only passages that appear in the champion lists of the query words. This is faster but may miss a good passage. We tried r = 50 and r = 200 and measured speed against quality: **r = 50 cut the median query time from 0.237 ms to 0.080 ms** (about three times faster) and left nDCG@10 unchanged at 0.560 (MAP 0.455 to 0.450); r = 200 gave no useful speed-up.

<div class="callout code"><b>CODE:</b> <code>index/tfidf.py</code>; measured by <code>eval/speed_quality.py</code> and <code>eval/cost_table.py</code>.</div>

### 3.5 BM25 (the standard strong baseline)

BM25 improves on TF-IDF in two ways. First, **term-frequency saturation**: the tenth occurrence of a word adds much less than the first. Second, **length normalisation**: a long passage should not win just because it has more words. The score is

> score(q, d) = Σ idf(t) × tf (k1 + 1) / ( tf + k1 (1 − b + b |d| / avgdl) )

k1 controls how fast the term-frequency gain saturates and b controls how strongly length is normalised. **Saturation example** (passage of average length, k1 = 0.6): tf = 1 gives a factor 1.00, tf = 3 gives 1.33, tf = 10 gives 1.51, and the factor can never exceed k1 + 1 = 1.6. We tuned k1 and b on the **10 tuning questions only** (grid of 30 combinations) and got k1 = 0.6 and b = 0.9 (dev nDCG@10 0.795, against 0.758 for the textbook 1.2 and 0.75). The surface was flat near its top, so we do not read the exact optimum as meaningful.

<div class="callout demo"><b>DEMO:</b> Search tab, choose bm25, open "Why this score" under a result: a bar chart shows how much each query word contributed.</div>

<div class="callout code"><b>CODE:</b> <code>ranking/bm25.py</code> (with an <code>explain()</code> function), <code>eval/tune_bm25.py</code>.</div>

### 3.6 Dense retrieval

A pre-trained **sentence encoder** (all-MiniLM-L6-v2) turns each passage and the query into a 384-number vector so that texts with similar meaning have similar vectors; passages are ranked by cosine similarity. It finds paraphrases that share no words with the question. It can be weaker on exact names and rare terms, which is why we combine it with BM25. The encoder is a pre-trained model used as a component; the search code around it is ours. The passage vectors are computed once and cached.

### 3.7 Hybrid search by reciprocal rank fusion (RRF)

BM25 scores and cosine scores live on different scales, so we do not add them. Reciprocal rank fusion uses only **ranks**: each passage gets 1 / (60 + rank) from each list and the contributions are summed. **Worked example:** passage A is ranked 1st by BM25 and 3rd by the dense model: 1/61 + 1/63 = 0.0164 + 0.0159 = 0.0323. Passage B is 2nd by BM25 and 1st by dense: 1/62 + 1/61 = 0.0161 + 0.0164 = 0.0325. So B ends up above A: it is good in both lists. (Cormack et al., 2009.) We also implemented CombSUM and CombMNZ (Fox and Shaw, 1994) and compared them in the ablations.

<div class="callout code"><b>CODE:</b> <code>ranking/dense.py</code>, <code>ranking/fusion.py</code>.</div>

### 3.8 Cross-encoder reranking

The first stage looks at the query and each passage separately. A **cross-encoder** reads the query and a passage **together**, so it can use word-by-word interactions and is more accurate, but it is too slow to run on 11,964 passages. We run it only on the top 30 candidates and keep five (model: ms-marco-MiniLM-L-6-v2). In the sidebar of the web app this is the tick box "Cross-encoder reranker".

### 3.9 Pseudo-relevance feedback (Rocchio)

Assume the top few results are relevant, take the most characteristic words from them and add them to the query (Rocchio, 1971). It can fix a query that uses different words from the filing. We tried nine settings (number of feedback passages 3, 5 or 10; number of added terms 5, 10 or 20); **none helped** on our collection (the best was about +0.004 nDCG@10, the others slightly negative).

### 3.10 Company-name matching with spelling tolerance (lecture on edit distance)

Company names, tickers and aliases are the dictionary. For a word in the question, the module first builds a **bigram shortlist**: the word is cut into pairs of letters, and dictionary entries are ranked by the **Jaccard overlap** of their bigram sets (cheap, so we do not compare against everything). Candidates are then confirmed with **Levenshtein distance** (minimum insertions, deletions and substitutions): at most two edits, and at most one edit for short strings such as tickers. **Example:** "Telsa" and "Tesla" differ by swapping two letters, which costs two edits, so it is accepted. The same rule explains the one miss: "MSTF" for MSFT also costs two edits, which is above the limit of one for a four-letter ticker. Guards stop common words from being turned into companies. This is why 39 of 40 test typos were fixed.

<div class="callout demo"><b>DEMO:</b> System tab, "typo lab": type <code>Micorsoft</code> (fixed, 2 edits) and then <code>MSTF</code> (the known miss).</div>

<div class="callout code"><b>CODE:</b> <code>query/entities.py</code>; evaluated by <code>eval/entity_eval.py</code>.</div>

### 3.11 Table chunks and the metadata prefix

Because every table is one chunk, a question such as "revenue by segment" can be answered from one passage with its labels. We also tested adding the company, year and section to the text of each passage before indexing ("metadata prefix"). It was the largest preprocessing effect we measured (+0.083 nDCG@10 in its best variant, interval +0.011 to +0.170), while stemming hurt (−0.066) and dropping table chunks or the company filter made no measurable difference.

### 3.12 Retrieval-augmented generation and caching

RAG means: retrieve text first, then let a language model write the answer from it (Lewis et al., 2020). Our prompt tells the model to use only the five passages, cite every claim and otherwise reply exactly INSUFFICIENT EVIDENCE. The call uses temperature 0 and is **cached on disk** under a hash of (model, prompt, parameters). That makes the demo repeatable, lets it run **offline**, and keeps the number of paid calls low. The answer model is Google's Gemini (`gemini-3.5-flash-lite`), chosen because larger models allow only about 20 free requests a day.

### 3.13 The verifier and the refusal rule

The verifier is deterministic. It tests (1) every cited [Pn] exists among the five passages; (2) the answer cites something; (3) every sentence carries a citation (abbreviations such as "U.S." or "Mr." do not split sentences); (4) every number in the answer occurs in a cited passage after normalising dollar signs, commas, percentages and units ("(in millions)" tables are honoured). **What it cannot do:** it checks *presence*, not *meaning*, so a number could be in the right passage but attached to the wrong claim, and it cannot detect an omission (in one test question, Walmart's segments, both retrieval-based answers left a segment out). On our data it flagged none of the 55 answered outputs, so it never demonstrated catching a real error (its unit tests do).

The refusal rule is: decline when the model replies INSUFFICIENT EVIDENCE. We also built a score rule (decline if the retrieval score is below a threshold tau) and tuned tau on the tuning questions; it came out as 0, which switches the rule off, because the fusion score does not separate answerable from unanswerable questions. In the sidebar, "Abstention threshold tau" is that control.

### 3.14 Company graph: PageRank and HITS (lectures on link analysis)

We draw an arrow from company A to company B if A's filing mentions B in at least two passages (strict, case-sensitive whole-word matching of names of four or more letters, so tickers such as BA or KO do not create false links). PageRank scores a company by the scores of the companies that mention it:

> PR(v) = (1 − d) / |V| + d × Σ PR(u) / outdeg(u), with damping d = 0.85

computed by power iteration; HITS gives each company a hub and an authority score. The graph has 20 nodes and 14 edges. **PageRank ordered the companies almost exactly like the plain count of incoming mentions** (Spearman 0.933), so on this small graph it adds little (hypothesis H10 not supported). HITS is almost degenerate here (only three companies get a non-zero authority score).

### 3.15 The page-image track and what the report says

The optional vision track was gated in the proposal (it needs a vision-language model, normally on a GPU). The submitted report states that only the MaxSim scoring was built and the track was not run. The next subsection describes the small demonstration that was added afterwards.

### 3.16 The page-image demo (the Pages tab), added after the report

The report says the page-image track was not built. After that we built a **small demonstration** so that you can show a working Pages tab. What it is, in plain words:

* **The idea (ColBERT / ColPali, late interaction).** Instead of extracting text from a page, a vision-language model looks at the **picture of the page** and turns it into many small vectors (one per image patch, here about 875 per page). A question is turned into one vector per word. The **MaxSim** score of a page is: for each question vector, take its best-matching page vector, then add these maxima. The page with the highest total is returned. This is the late-interaction scoring we wrote ourselves in `src/trademesh/vision/maxsim.py`; only the vectors come from the pre-trained model.
* **What we ran.** 39 pages from four filings (Apple, Microsoft, Tesla, Walmart), rendered to images; 8 hand-written questions, mostly about tables (net sales by product, revenue by segment, income statement, balance sheet, shareholder-return table). The model is `vidore/colSmol-256M` (256 million parameters), run on the **CPU** because the laptop's graphics card has only 4 GB and the Python environment has the CPU build of PyTorch; it took about 38 seconds per page.
* **A fair comparison.** For the same questions and pages we also ran **BM25 over the text extracted from the pages**. A result counts as correct when its page contains the answer (a page matching the question's keyword rule).
* **Result.** MaxSim over page images: the first result was a correct page for **8 of 8** questions and a correct page was in the top three for **8 of 8**. BM25 over extracted text: **7 of 8** and **8 of 8**.

| Question | Question (shortened) | Rank of first correct page, vision | Rank, text BM25 |
|---|---|---|---|
| V1 | Which page shows Apple's net sales by product category: iPhone, Mac, iPad, Wea... | 1 | 1 |
| V2 | Where is the table of Apple's five-year cumulative total shareholder return co... | 1 | 1 |
| V3 | Microsoft revenue and operating income by segment: Productivity and Business P... | 1 | 1 |
| V4 | Microsoft income statement with net income and earnings per share | 1 | 1 |
| V5 | Tesla statement of operations: automotive sales, regulatory credits and energy... | 1 | 1 |
| V6 | Tesla balance sheet with total assets and total liabilities | 1 | 1 |
| V7 | Walmart net sales and operating income by segment: Walmart U.S., Walmart Inter... | 1 | 1 |
| V8 | Walmart consolidated balance sheet with total assets | 1 | 2 |

* **How to read it honestly.** This is eight questions on 39 pages written by us, so it shows that the approach works on this kind of page; it does not show that vision is better or worse than text in general (that would need many more questions and the full corpus). The pages are real filing pages, but the filings' charts are separate image files that we did not download, so a chart shows as an empty frame; the numbers around it are real. The demo is **not in the submitted report**, which still says the track was not built; say that it was added afterwards as a proof of concept.

@@FIG:docs/report/figures/app_pages_top.png|120|Figure 12. The Pages tab: page-image search (MaxSim) against text search (BM25) on the same 39 pages, then the best four pages for each question.@@

<div class="callout demo"><b>DEMO:</b> Pages tab. The four tiles at the top compare page-image search with text search. Below, each question shows the four best pages as pictures; a page marked "[correct page]" contains the answer.</div>

<div class="callout code"><b>CODE:</b> <code>eval/vision_prepare.py</code> (choose and render pages, write questions), <code>eval/vision_encode.py</code> (the pre-trained vision model), <code>eval/vision_score.py</code> (our MaxSim and the BM25 baseline), <code>src/trademesh/vision/maxsim.py</code>, results in <code>data/vision/results.json</code>, instructions in <code>docs/vision_demo.md</code>.</div>

## 4. How we evaluated, and how to explain the numbers

### 4.1 The test collection

We wrote **30 information needs** (a question plus a short keyword query) in three types: **entity** (about one named company), **topical** (across companies, for example "supplier concentration risk") and **terminology** (a defined term or a figure in a table; five of them need a table). **10 needs are for tuning** (every choice such as BM25's k1 and b is made on these) and **20 are the test needs**, scored once after the code was frozen; a lock file stops a second run from overwriting the result. We also wrote 10 **unanswerable** questions (4 tuning, 6 test) such as "What was Apple's net sales in fiscal year 2031?", and 40 **misspelt company names** for the spelling module.

### 4.2 Pooling and relevance labels

We cannot judge all 11,964 passages for every question. **Pooling** (a lecture topic): for each question we merge the top 8 results of every system, remove duplicates, shuffle them and hide which system found what. The pools hold 11 to 30 passages per question (mean 19.6). A language model graded each pooled passage 0 (not relevant), 1 (partly) or 2 (answers the need) without seeing the system; a second pass graded 91 of them again (weighted kappa 0.785 between the two passes). **Be upfront: no human graded a relevance label.** Pooling has a known bias in favour of the systems that built the pool (Zobel, 1998); we ran checks (the share of each system's top 10 that was judged, bpref, and leave-one-system-out), which are in the detailed report.

### 4.3 The metrics (lecture on evaluation)

| Metric | Meaning |
|---|---|
| P@10 | share of the top 10 results that are relevant |
| R@10 | share of all relevant passages (in the pool) that appear in the top 10 |
| F1@10 | the harmonic mean of the two: 2PR / (P + R) |
| MAP | the mean over questions of the average precision at each relevant result |
| MRR | the mean of 1 / (rank of the first relevant result) |
| nDCG@10 | rewards putting the most relevant passages first; 1.0 is a perfect ordering |
| Accuracy | used for spelling (39 of 40) and abstention (correct decisions out of all items) |

**nDCG example:** a question with two relevant passages. If the system returns them at ranks 1 and 3, DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5. The ideal order (ranks 1 and 2) gives 1 + 1/log2(3) = 1.631. So nDCG = 1.5 / 1.631 = **0.92**.

### 4.4 Why a bootstrap, and what "clear" means

With only 20 test questions, a classic significance test is unreliable. For two systems we take the difference in nDCG@10 on each question, resample the 20 questions with replacement 10,000 times, and read the **95% interval** of the mean difference from the resamples. We call a difference **clear** only if the interval excludes zero (decision rules fixed in `eval/hypotheses.py` before the test run). Otherwise it is "unclear", and we say so instead of reading a trend into noise.

### 4.5 The 11 hypotheses (written before the test run) and what happened

| H | Hypothesis | Outcome |
|---|---|---|
| H1 | BM25 beats TF-IDF | **Supported** (+0.139 nDCG@10) |
| H2 | BM25 ties dense on exact-term questions; dense wins on topical ones | Unclear (only 6 or 7 questions per type) |
| H3 | Hybrid beats BM25 and dense | Beats BM25 (+0.085, interval +0.019 to +0.155); tied with dense (+0.002) |
| H4 | Reranker improves hybrid | Unclear (+0.044, interval −0.012 to +0.113) |
| H5 | Table chunks and numbers help table questions | Not shown (no effect over all questions; the table-only subset was not computed) |
| H6 | Pseudo-relevance feedback helps | No (tied with BM25) |
| H7 | Champion lists are faster at little cost | Yes for r = 50 |
| H8 | Answers with retrieval beat answers without | **Supported** for the best retriever (+0.625); weak for BM25 (+0.325) |
| H9 | Hybrid + reranker answers beat BM25 answers | **Supported** (+0.300) |
| H10 | PageRank differs from counting mentions | No (Spearman 0.93) |
| H11 | Page-image retrieval beats text | Not tested in the report; the later 8-question demo is only a proof of concept (section 3.16) |

### 4.6 The human rating study (and the one correction we disclose)

Two of us (Shashwat and Ruchir) rated the three kinds of answers to the 20 test questions (60 answers) **blind**: the order of the three answers was shuffled and no system name was shown. Each wrote a reference answer from the filings first, then rated each answer 0 (wrong, invented or no answer), 1 (partly correct or incomplete) or 2 (correct and complete). **Kappa** measures agreement beyond chance: ours is 0.39 (moderate); the raters agree exactly on 41 of 60 answers and never differ by more than one point. A language-model rater scored much higher than both humans (1.43 against about 0.9 on average), so the human ratings are the main result. **Disclosed correction:** for one question (T10) the first reference said the filings contain no answer; we then found that Amazon's and Microsoft's own filings do answer it, and the reference was rewritten after the first scoring. In the single-rater scoring this changed one comparison (+0.25, unclear, to +0.40, clear), so both results are reported. Ruchir's independent ratings of that question match the corrected ones.

## 5. Results, with what to say

### 5.1 Retrieval

| System | P@10 | R@10 | F1@10 | MAP | nDCG@10 |
|---|---|---|---|---|---|
| TF-IDF | 0.530 | 0.395 | 0.428 | 0.455 | 0.560 |
| BM25 | 0.615 | 0.512 | 0.519 | 0.574 | 0.699 |
| Dense | 0.640 | 0.606 | 0.566 | 0.606 | 0.782 |
| Hybrid (BM25 + dense) | 0.670 | 0.604 | 0.584 | 0.678 | 0.784 |
| **Hybrid + reranker** | **0.705** | **0.658** | **0.621** | **0.693** | **0.828** |

Other systems: Boolean 0.364, TF-IDF with champion lists 0.560, BM25 + feedback 0.679, BM25 + reranker 0.776.

@@FIG:docs/report/figures/fig_ndcg_systems.png|130|Figure 4. nDCG@10 of the ten setups on the 20 test questions; the right column is the difference from BM25 with its 95% interval.@@

@@FIG:docs/report/figures/fig_pr_curves.png|115|Figure 5. Precision against recall. A curve that stays higher is better.@@

<div class="callout say"><b>SAY:</b> "Hybrid retrieval with the reranker is best at 0.828, ahead of BM25 at 0.699 and TF-IDF at 0.560. The differences that are clear are BM25 over TF-IDF, hybrid over BM25, and hybrid with the reranker over BM25. Hybrid and dense are tied, and the reranker's extra gain over hybrid is not certain with only 20 questions." It was best for every question type; BM25 was weakest on questions about one named company (0.58 against 0.71).</div>

<div class="callout demo"><b>DEMO:</b> Evaluation tab, Retrieval: the bars, the metric table, the precision-recall chart and the by-need-type table.</div>

### 5.2 What helped and what did not

@@FIG:docs/report/figures/fig_ablation_s.png|85|Figure 6. One change at a time, as a difference in nDCG@10 from plain BM25 with its 95% interval. Dark means the interval excludes zero.@@

Adding the company, year and section to each passage helped most; stemming hurt; table chunks, the company filter and feedback made no measurable difference. Fusion with RRF k = 60 gave +0.085 and k = 10 gave +0.121 (k = 60 is the standard value we fixed in advance; the k = 10 result is an after-the-fact observation).

### 5.3 Spelling correction

39 of 40 misspelt names were fixed: deletions 15 of 15, substitutions 10 of 10, transpositions 4 of 4, insertions 2 of 2, spacing mistakes 3 of 3, ticker mistakes 5 of 6. The only miss is MSTF for MSFT (two edits). The 40 typos were written by us, so this measures the typo types we thought of, not real user typing.

@@FIG:docs/report/figures/fig_entity_accuracy.png|105|Figure 7. Spelling correction by type of mistake.@@

### 5.4 Answer quality

| Answers made with | Mean of both raters | Shashwat | Ruchir | Model-assisted rater |
|---|---|---|---|---|
| Without retrieval (closed-book) | 0.575 | 0.70 | 0.45 | 1.175 |
| BM25 + answer | 0.90 | 0.80 | 1.00 | 1.325 |
| **Hybrid + reranker + answer** | **1.20** | 1.20 | 1.20 | 1.70 |

The best retriever beat answers without retrieval by +0.625 (95% interval +0.350 to +0.900) and beat BM25-based answers by +0.300 (interval +0.125 to +0.500). BM25-based answers beat no retrieval only weakly (+0.325, interval +0.050 to +0.575), and that comes mostly from one rater (Ruchir +0.55, Shashwat +0.10). The automatic checks were clean: citations that support their sentence 31 of 32 for the best retriever (counting only citations both raters checked), and every number found in the cited text.

@@FIG:docs/report/figures/fig_rag_ratings.png|115|Figure 8. Average answer rating (0 to 2) for the humans and for the language-model rater.@@

<div class="callout say"><b>SAY:</b> "Answers built on the best retriever scored 1.2 out of 2 against 0.575 for the same model without the filings. The two human raters agree on 41 of 60 answers, kappa 0.39, and order the three systems the same way. We treat the human ratings as the main result because the language-model rater was more generous than both humans. Limits: both raters are on our team."</div>

<div class="callout demo"><b>DEMO:</b> Ask tab, Figure / segments, then switch on the <b>Closed-book</b> toggle to see the same question answered without the filings (older fiscal-year figures, no citations, a red "unverified" panel). Evaluation tab, RAG answers shows the human-rated panel.</div>

### 5.5 Knowing when not to answer

| Rule | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Retrieval-score threshold | 0.38 | 0.83 | 0.53 | 0.65 |
| **The model's own verdict** | **1.00** | **1.00** | **1.00** | **1.00** |

(26 test items: 6 unanswerable, positive class = should refuse.) The model's verdict declined 6 of 6 unanswerable questions and none of the 20 answerable ones. **Be honest:** the unanswerable questions are easy (clearly outside the filings), and the refusal relies on the model's own verdict, so this shows that the idea works on clear cases, not that it is robust in general.

@@FIG:docs/report/figures/fig_abstention.png|105|Figure 9. Retrieval-score rule against the model's own verdict.@@

<div class="callout demo"><b>DEMO:</b> Ask tab, <b>Trap / future year</b> (Apple's net sales in fiscal year 2031): a red "Insufficient evidence" panel and "Decision: Abstain".</div>

### 5.6 Company graph

@@FIG:docs/report/figures/fig_graph_net_s.png|85|Figure 10. Who mentions whom (node size is PageRank).@@

<div class="callout demo"><b>DEMO:</b> Graph tab: PageRank bars (AAPL first, XOM second), the node graph, and the tile "0.93: PageRank vs in-degree Spearman".</div>

### 5.7 Comparison with published work

| Published claim | Paper | Our result | Same? |
|---|---|---|---|
| BM25 is a strong baseline | BEIR (Thakur et al., 2021) | Dense was ahead of BM25 here, BM25 not ahead | No |
| Dense wins on paraphrased questions | DPR (Karpukhin et al., 2020) | Same direction, but unclear | Partly |
| Fusing rankings helps | RRF (Cormack et al., 2009) | Hybrid beats BM25, ties dense | Yes |
| Cross-encoder reranking helps | Nogueira and Cho (2019) | Same sign, interval includes 0 | Partly |
| Retrieval matters for filings | FinanceBench (Islam et al., 2023) | Best retriever +0.625 over closed-book | Yes |
| Page images help on visual pages | ColPali (Faysse et al., 2024) | Not tested | n/a |

We compare directions, not absolute numbers, because the corpora differ.

## 6. The demo: how to run it and what to show

### 6.1 Before you present (the day before, and 15 minutes before)

1. Plug in, connect to the internet, close heavy programs (Docker Desktop makes the laptop slow).
2. Free port 8501 (close the rating tool or any other Streamlit window).
3. Check that `.env` has `LLM_PROVIDER=gemini` and your `GEMINI_API_KEY` (never show the file on screen).
4. Check the cache works offline; every demo question must be answered from the cache.
5. Start the app **at least 5 minutes before** you present. A warm start takes about a minute; a cold first start after a restart took about four minutes because it loads the dense model. Open the page and wait until the tab bar appears; do not reload while it says "Building the search indexes".
6. In the sidebar check: retriever **hybrid**, **Cross-encoder reranker ticked**, tau 0.00, lambda 0.00. These are the reported settings.
7. Do one dry run of the first four scenes and record the screen (Win+Alt+R) as a backup.

### 6.2 Commands (PowerShell)

```
cd C:\Users\shash\Downloads\IRFinalProject

# check that every demo question is answered from the cache (no network needed)
$env:LLM_OFFLINE="1"; .\.venv\Scripts\python.exe eval/demo_warm.py; Remove-Item Env:LLM_OFFLINE

# start the app, then open http://localhost:8501
.\.venv\Scripts\python.exe -m streamlit run src/trademesh/app/streamlit_app.py
```

* **No internet at the venue:** set `$env:LLM_OFFLINE="1"` before starting. The four example buttons are exactly the cached questions, so they still work; a brand-new question would show "not in the answer cache".
* **Port busy:** add `--server.port=8502` and open that address.
* **Docker (packaging only):** `docker compose up --build`, or for the full setup `$env:INSTALL_ML="1"; docker compose up --build`.
* **Tests:** `.\.venv\Scripts\python.exe -m pytest -q` (137 tests, offline).

### 6.3 The page and its tabs

The page, top to bottom: a hero with two buttons ("Ask with a typo" and "Compare two companies"), a company strip, status chips (corpus, retrievers, LLM, passages, companies, LLM calls live / from cache), then six tabs.

| Tab | What it shows |
|---|---|
| **Ask** | The question box and four example buttons; toggles Closed-book and Compare; a pipeline strip of seven chips; the answer with P-tags; evidence cards; the trust panel |
| **Search** | Query box, up to three retrievers side by side, company filter, top K; highlighted words, hits by 10-K item, "Why this score" |
| **Graph** | Company mention graph: PageRank bars and a node graph |
| **Pages** | Small page-image demo: tiles comparing page-image search (MaxSim) with text search (BM25), then the four best page pictures for each of 8 questions, with the correct page marked |
| **Evaluation** | Three sub-tabs: Retrieval, RAG answers, Abstention, all read from the locked result files |
| **System** | Readiness checklist, headline tiles, and the typo lab |

The sidebar has the controls for the answer pipeline: retriever, cross-encoder tickbox, abstention threshold tau and section-prior lambda.

@@FIGROW:docs/report/figures/app_eval.png|0,60,1140,1030|docs/report/figures/app_graph.png|0,0,1140,1090|Figure 11. Left: the Evaluation tab (retrieval results read from the result files). Right: the Graph tab.@@

### 6.4 The live script (about 8 minutes)

| Time | Click | Say |
|---|---|---|
| 0:00 | Show the home screen. | "11,964 passages, 20 companies. Every claim carries a citation, a checker verifies the numbers, and it says so when the evidence is thin. The search core is written from scratch." |
| 0:30 | **Ask with a typo** ("Telsa"). | "Our own spelling module, a bigram shortlist plus Levenshtein distance, corrected Telsa to Tesla; it fixed 39 of 40 test typos. The tag P1 is the passage that backs the claim; open it." Hover the pipeline chips. |
| 1:30 | Ask tab, **Figure / segments**. | "The verifier checks every number against the cited passage. An invented figure would show red here." |
| 2:15 | Switch **Closed-book** on, then off. | "Same model without the filings: fluent but out of date and uncheckable. In our blind rating, retrieval plus reranking scored 1.20 out of 2 against 0.575." |
| 3:00 | **Compare two companies**. | "Naming two companies makes one search per company and merges the results." |
| 3:45 | **Trap / future year**. | "It declines instead of guessing. Our unanswerable questions are easy, and a score threshold did not work, so we use the model's own verdict." |
| 4:30 | Search tab: `supplier concentration`; choose bm25, tfidf, hybrid. | "Our retrievers side by side. Why this score shows each word's contribution." Optionally pick **boolean** and try a phrase or `cyber*`. |
| 5:15 | Evaluation tab. | "Hybrid with reranking 0.828 against 0.699 for BM25. Two of us rated 60 answers blind." State the limits. |
| 6:15 | Graph tab. | "Who mentions whom. PageRank correlates 0.93 with plain counting, so it adds little." |
| 6:45 | System tab, typo lab: `Micorsoft`, then `MSTF`. | "Readiness is honest about what is wired up." Then open the Pages tab (section 3.16): "This is a small proof of concept that we added after the report: a vision model reads page pictures and our MaxSim ranks them." |

**Do not type live:** possessive company names ("Microsoft's", "Apple's"): the spelling module skips them when the possessive form already occurs in the filings, so the company filter is not applied (a known defect that we disclose; it affected three company-naming questions in the test). **If something breaks:** play the backup video and narrate; if an answer takes very long, the client is waiting out the provider's rate limit, so switch to offline mode.

### 6.5 Reading every screen (what you are showing and what it means)

**Search tab.** It stays empty ("No query") until you type a query. You choose up to three retrievers and the page shows them side by side.

* `boolean`: exact matching with AND / OR / NOT, quoted phrases and wildcards; passages that match are not ranked by relevance. Use it to show precise control.
* `tfidf`: ranks by how often a word occurs in a passage and how rare it is in the collection (cosine similarity). The classic baseline.
* `tfidf-champ-50` / `tfidf-champ-200`: the same, but only the top 50 or 200 passages per word are scored. Same quality here, faster.
* `bm25`: TF-IDF improved by saturating repeated words and normalising for passage length. The strong keyword baseline.
* `bm25+prf`: BM25 whose query is expanded with words from its own top results. It did not help here.
* `dense`: compares meaning with sentence vectors, so it can match a paraphrase that shares no words.
* `hybrid`: merges the BM25 and dense rankings by rank (reciprocal rank fusion).
* The Company box restricts results to one company; Top K sets how many passages each column shows. The "hits by 10-K item" chart tells you which section of the filings the matches come from (for example risk factors against financial statements). "Why this score" lists how much each query word added to the BM25 score of a passage.

**Sidebar controls (they affect the Ask tab, not the Search tab).**

* **Retriever for answers:** which retriever supplies the five passages that the answer model sees. The reported setting is hybrid.
* **Cross-encoder reranker (tickbox):** the first stage scores a question and a passage separately. The cross-encoder reads them **together**, which is more accurate but too slow for 11,964 passages, so it only rereads the top 30 candidates and keeps five. Ticked is the best setup (nDCG@10 0.828).
* **Abstention threshold tau:** if the best retrieval score is below tau, the system declines to answer. We tuned tau on the tuning questions and it came out 0, which switches this rule off, because the retrieval score did not separate answerable from unanswerable questions (the score rule had precision 0.38). The model's own "INSUFFICIENT EVIDENCE" reply does the refusing instead. In the demo you can raise tau step by step to watch a normal question start to be declined on score alone; try it in your dry run first, because the right value depends on the score scale.
* **Section prior lambda:** adds a small bonus to passages from sections that usually hold answers: final score = base score + lambda x weight of the 10-K section (risk factors and management discussion weigh most, legal and exhibit sections least). It helped by at most +0.019 on the tuning questions, which is noise, so it is 0 in all reported results. Raising it to 0.1 gives a small reordering.

**Evaluation tab (Retrieval, RAG answers, Abstention).** The bars are nDCG@10 for each of the ten retrievers on the 20 locked test questions; the table lists P@5, P@10, MAP, MRR, nDCG@10, bpref and judged@10 (the share of each system's top 10 that has a relevance label); the precision-recall chart shows precision as recall rises. The caption "LLM-assisted, no human validation" refers to the relevance labels. The RAG sub-tab shows the human ratings first and the language-model ratings for comparison; the Abstention sub-tab shows the score rule against the model's own verdict.

**System tab.** The first row (Query, Retrieve, Rerank, Context, LLM, Verify, Abstain) is the pipeline reminder: entities and spelling, then BM25 / dense / hybrid, then the cross-encoder, then the top five passages as [P1] to [P5], then the language model at temperature 0 with a cache, then the citation and number check, then the refusal decision. The Readiness row checks what actually loaded: the corpus (11,964 passages), the retrievers, the reranker, the language model and the company graph. A red FLAG on a row means that item is not available; "Page images" shows its state honestly.

**Results tiles.** 1.20 is the mean human answer rating (0 to 2) for the best setup, against 0.57 without retrieval (0.575 exactly); 97.5% is the share of misspelt company names fixed (39 of 40), with a bar per type of typo; 6/6 is the unanswerable test questions declined, with 0 of the 20 answerable ones declined wrongly; 0.828 is the best retrieval score against 0.699 for BM25. **Missed: MSTF > MSFT.** The only typo we failed on is the ticker MSTF for MSFT, a swapped-letter ticker. Swapping two letters costs two edits in Levenshtein distance, and four-letter tickers allow only one edit, so the matcher refuses it. A transposition-aware edit distance would fix it; we left the code unchanged so that the reported results stay reproducible.

## 7. Report, demo and code: one map

Section numbers are those of the paper (Roman numerals). Code paths are relative to the repository root.

| Report section | What it says | Where in the demo | Code and results |
|---|---|---|---|
| I Introduction | Problem, importance, contributions | Home screen, tagline | `README.md` |
| II Related work (Table I) | Papers used for each part | (slide) | `docs/report/parts/10_references.md` |
| III Objective | RQ1 to RQ5, 11 hypotheses | Evaluation tab; Graph tab; Pages tab for RQ5 | `eval/hypotheses.py`, `eval/out/hypotheses_test.json` |
| IV Proposed model | Offline build and six online steps | Ask tab pipeline chips | `rag/pipeline.py`, `app/streamlit_app.py` |
| V-A Ten setups | The retrieval configurations | Search tab retriever box; sidebar | `retrievers.py`, `index/`, `ranking/` |
| V-B From-scratch search | Index, skips, Boolean, TF-IDF, BM25 | Search tab, "Why this score"; boolean | `index/positional.py`, `boolean.py`, `wildcard.py`, `tfidf.py`, `ranking/bm25.py` |
| V-C Modern retrieval | Dense, RRF, reranker, feedback | Sidebar hybrid and reranker | `ranking/dense.py`, `fusion.py`, `rerank.py`, `prf.py` |
| V-D Names, graph, answers | Spelling, compare, PageRank, verifier | Ask (typo, compare, trust panel); Graph; System typo lab | `query/entities.py`, `graph/mentions.py`, `rag/prompt.py`, `verifier.py`, `abstain.py`, `llm.py` |
| V-E Setup and software | Models, versions, seeds | System tab readiness | `requirements*.txt`, `Dockerfile` |
| V-F Evaluation protocol | Pooling, metrics, bootstrap, ratings | Evaluation tab | `src/trademesh/evaluation/`, `eval/run_retrieval.py`, `eval/human_*.py` |
| VI-A Data | Corpus and test collection | Home chips; System tiles | `data/passages.jsonl`, `data/needs/`, `data/judgments/qrels.csv` |
| VI-B Retrieval results | nDCG, P/R/F1, curves | Evaluation, Retrieval | `eval/out/retrieval_test.json` |
| VI-C Ablations and hypotheses | What helped | (report figures) | `eval/ablation.py`, `eval/out/ablation_test.json` |
| VI-D Spelling | 39 of 40 | System typo lab | `eval/entity_eval.py` |
| VI-E Answer quality | 1.20 vs 0.575, kappa 0.39 | Closed-book toggle; Evaluation, RAG answers | `eval/rag_run.py`, `eval/human_score.py`, `eval/human/raters/` |
| VI-F Abstention | 6 of 6 declined | Trap / future year; Evaluation, Abstention | `rag/abstain.py`, `eval/abstention_eval.py` |
| VI-G Company graph | 14 links, Spearman 0.93 | Graph tab | `graph/mentions.py` |
| VI-H Comparison with papers | Agrees on fusion; BM25 not ahead of dense | (slide) | `docs/report_B/10_paper_comparison.md` |
| VI-I to K Discussion, failures, cost | Tables are hardest; "'s" defect | System tab | `docs/report_C/10_rag_results.md`, `eval/out/cost_table.json` |
| VII Conclusions and limitations | Small set, model labels, team raters; report says page images not built | Pages tab (small demo added afterwards) | `vision/maxsim.py`, `eval/vision_*.py`, `data/vision/results.json` |

## 8. Novelty and honest limits

### 8.1 What is new (say it modestly)

We do not claim a new model. The contribution is the combination and the evaluation:

1. **A from-scratch classic core next to the modern parts**, compared like-for-like (ten setups, same questions, same judgments) on financial passages.
2. **A checkable answer layer.** A deterministic verifier for citations and numbers, plus a refusal rule, with a negative result we report openly: a threshold on the retrieval score does not separate answerable from unanswerable questions (precision 0.38), so the system relies on the model's own verdict.
3. **Finance-aware handling.** Every table is one chunk; a typo-tolerant company matcher written from scratch; compare mode that runs one search per company.
4. **Evaluation discipline.** Eleven hypotheses written before one locked test run, paired bootstrap intervals, pool-bias checks, blind human ratings by two raters with their agreement, and honest reporting of unclear results and one disclosed correction.
5. **A company mention graph**, with the honest finding that PageRank adds little over counting mentions.
6. **Reproducible and runnable.** Docker, an offline answer cache, 137 automatic tests, and every number in the report regenerated by a script.

### 8.2 What we did not do, and what the results do not prove

* The **page-image track** (RQ5, H11): the report says it was not built. Afterwards we added a small demonstration (8 questions, 39 pages, a 256-million-parameter model on the CPU). It is a proof of concept, not a result for the report.
* Relevance labels were made by a **language model**; no human checked them.
* Both answer raters are **on our team**; one checked only 48 of the 95 citations; kappa is moderate (0.39).
* The test set is **small** (20 questions), so many intervals are wide and many comparisons are "unclear".
* **One small answer model**; unanswerable questions are **easy**; the perfect refusal result should not be over-read.
* Company names with **"'s"** are not recognised by the spelling module when the possessive form occurs in the filings.
* Only the **latest filing** of each company is indexed, so trend questions across years are out of scope.
* It is a research prototype, **not financial advice**.

## 9. Questions you may be asked

**What is built from scratch and what is pre-trained?** Built by us: the positional index with skip pointers, Boolean/phrase/wildcard search, TF-IDF, BM25, champion lists, feedback, spelling correction, rank fusion code, the verifier, the metrics, pooling and the bootstrap. Pre-trained and used as parts: the sentence encoder and the cross-encoder, and the answer model (Gemini).

**Why did you not just use a language model on the whole filing?** The filings are far longer than a model reads reliably, an unaided model can invent or misquote figures, and it does not point to a source. Retrieval narrows the input to five passages and makes the answer checkable.

**How does BM25 differ from TF-IDF?** It saturates term frequency (k1) and normalises for passage length (b). We tuned k1 = 0.6 and b = 0.9 on the tuning questions only; BM25 beat TF-IDF by 0.139 nDCG@10 (clear).

**Why does hybrid help?** Exact words (company names, defined terms) favour BM25, paraphrased questions favour the dense model, and fusing ranks uses both signals without calibrating scores. Hybrid beat BM25 clearly and tied dense.

**Why is the reranker only on the top 30?** It reads query and passage together, which is accurate but slow; running it on everything is impractical. Its gain over hybrid was +0.044 with an interval that includes zero, so we call it unclear.

**How do you know the answers are right?** We do not claim they always are. We check that every citation exists and every number appears in the cited text, and we rate answer quality blind against reference answers. The checker tests presence, not meaning.

**Why is tau zero?** The fusion score does not separate answerable from unanswerable questions (precision 0.38 for the score-only rule), so tau tuned on the tuning questions came out as zero and abstention uses the model's own verdict.

**Is a refusal rate of 6 of 6 impressive?** It is a good sign but not proof: the unanswerable questions are easy and the lower bound of the 95% interval for 6 of 6 is about 0.54.

**What is nDCG?** A graded measure that rewards ranking the most relevant passages first, normalised so 1.0 is perfect. Example: relevant passages at ranks 1 and 3 give 0.92.

**What is pooling and what is its weakness?** We judge only the union of the top 8 of every system. It saves effort but can favour systems that built the pool (Zobel, 1998), and a relevant passage nobody retrieved counts as not relevant. We checked with bpref, judged@10 and leave-one-system-out.

**Why a bootstrap?** With 20 questions a conventional test is unreliable; resampling questions gives an interval and lets us say "clear" only when it excludes zero.

**What is kappa and what does 0.39 mean?** Agreement beyond chance between two raters (1 is perfect, 0 is chance). 0.39 is moderate: the raters agree exactly on 41 of 60 answers and never differ by more than one point.

**Why is the human rating lower than the model rating?** The language-model rater was more generous: it gave the higher score whenever they differed. We think the human references were longer, so "complete" was a higher bar; we did not test this. That is why the human ratings are the main result.

**What happened with question T10?** The first reference said the filings contain no answer, but Amazon's and Microsoft's filings do answer it; the reference was rewritten after the first scoring and both results are reported. Ruchir's independent ratings match the corrected ones.

**How does the spelling module work, and why did MSTF fail?** Bigram shortlist (Jaccard overlap) then Levenshtein distance, at most two edits and one for short strings. Swapping two letters costs two edits, which is above the limit for a four-letter ticker. A transposition-aware distance would fix it.

**What is a champion list?** For each word we keep the top r passages by weight and score only those; r = 50 was about three times faster at almost no loss.

**What does the company graph tell us?** Little on this small graph: 14 links, PageRank almost the same as in-degree (Spearman 0.93), and HITS nearly degenerate. We report it as a negative result.

**Why was the vision track not in the report?** It was optional and gated (it needs a vision model, normally on a GPU) and was not run in time. We wrote and tested the MaxSim scoring and said so in the report. After submission we added a small demonstration with a small ColPali-family model on the CPU: 8 hand-written table questions on 39 pages from four filings, where page-image search put a correct page first for 8 of 8 questions and text search (BM25) for 7 of 8. It shows that the approach works; with so few questions it does not show which method is better.

**What would you do next?** More raters and test questions, human checks of the relevance labels, a fix for the possessive-name problem, harder unanswerable questions, several filing years and, if a GPU is available, the page-image track.

**How is it reproducible?** Pinned requirements (including the optional ML packages), an on-disk answer cache so the demo runs offline, Docker, 137 tests, and scripts that regenerate every result file; the test run is locked after one execution.

## 10. Glossary

| Term | Meaning |
|---|---|
| 10-K | The yearly report a listed US company files with the SEC |
| Passage | The retrieval unit: a text chunk of a 10-K section, or one whole table |
| Inverted index | Maps each word to the passages that contain it |
| Positional index | An inverted index that also stores word positions (enables phrase search) |
| Skip pointer | A shortcut in a postings list that speeds up intersection |
| TF-IDF | Weight by frequency in the passage and rarity in the collection |
| BM25 | A ranking function with term-frequency saturation and length normalisation |
| Champion list | The top r passages stored per word, for faster approximate search |
| Dense retrieval | Ranking by similarity of sentence vectors from a neural encoder |
| RRF | Reciprocal rank fusion: add 1 / (60 + rank) over the lists |
| Cross-encoder | A model that reads query and passage together to rerank candidates |
| Pseudo-relevance feedback | Expanding the query with words from the top results |
| RAG | Retrieval-augmented generation: retrieve text, then generate the answer from it |
| Closed-book | The same model answering without any retrieved passages |
| Verifier | The program that checks citations and numbers |
| Abstention | Declining to answer when the evidence is insufficient |
| Pooling | Judging only the union of the top results of all systems |
| nDCG, MAP, MRR | Ranking metrics (see section 4.3) |
| Bootstrap | Resampling questions to get a confidence interval |
| Kappa | Agreement between raters beyond chance |
| PageRank, HITS | Link-analysis scores (importance; hub and authority) |
| MaxSim | Late-interaction score: sum over query vectors of the best-matching page vector |

## 11. A slide outline (10 to 12 slides)

1. Title, team (names and roll numbers).
2. The problem: long filings, keyword search, unchecked language models.
3. The idea (search first, model second, checker last) and the contributions.
4. The pipeline diagram (Figure 1) and the app.
5. Built from scratch: index, TF-IDF and BM25 (the two formulas).
6. Modern retrieval: dense, rank fusion (with the A and B example), reranker.
7. Checkable answers: the flow diagram (Figure 2) and the Tesla example.
8. Data and test collection: 20 companies, 30 questions, pooling, the limits of the labels.
9. Retrieval results: Table and Figure 4.
10. Answer quality and refusal: 1.20 against 0.575, kappa 0.39, 6 of 6 declined, with the limits.
11. **Live demo** (section 6.4).
12. Limitations, future work, questions.

<div class="callout say"><b>CLOSING LINE:</b> "Everything runs from Docker or pip, every result regenerates from a script, and every number in the report comes from a file in the repository. We report what worked and what did not, including that many differences are unclear with 20 test questions."</div>

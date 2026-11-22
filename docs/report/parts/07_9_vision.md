## 7.9 Page-image retrieval (gated vision track): scoped out

The proposal contained an optional track that retrieves 10-K pages directly from page images, for questions whose answer sits in a chart or a laid-out table that text extraction flattens. It was gated: the track would be attempted only if a small go/no-go check passed, and cancelled and reported otherwise. **The track was not built, and no result for it (RQ5, H11) exists.** The Pages tab of the application and the readiness checklist state this.

**What is implemented.** The late-interaction scoring of ColPali-style retrieval (Faysse et al. 2024; the MaxSim operator of ColBERT, Khattab and Zaharia 2020). A query is a set of n token vectors q_1..q_n and a page a set of m patch vectors p_1..p_m, and

    MaxSim(q, page) = sum over i = 1..n of max over j = 1..m of <q_i, p_j>

Pages are ranked by this score by brute force (a few hundred pages need no vector database), ties broken by page id. The scoring function, a ranking helper and a writer for the results file the application reads are in `src/trademesh/vision/maxsim.py` and are covered by unit tests on hand-computed vectors.

**What is not implemented.** The page images, the vision-language encoder that turns pages and queries into vectors (`ColPaliEncoder` is a stub that says so) and the go/no-go check itself (embed five pages and test whether MaxSim returns the right page for three of five hand-written questions). The encoder needs a GPU for practical indexing and a multi-gigabyte model download; we did not set that up, so the check was not run (and so did not fail either).

**Why this is a sound omission.** (1) The track was optional and gated in the proposal, and the gate was not passed (the check was never run), so leaving it out is within the plan. (2) The text pipeline already indexes every table as a chunk, and the table-lookup questions are covered by the text-based experiments (with the weakness reported in section 8.12). (3) Adding an untested vision component, or reporting a number for it, would have been less honest than leaving it out; the claims in this report are limited to what was run. What the omission costs: the report cannot say anything about charts and images in 10-Ks, and H11 and RQ5 stay open.

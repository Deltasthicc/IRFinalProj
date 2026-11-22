### 9.3 Future work

1. A second human rater for the answer study (giving a human-to-human kappa) and a human check of a sample of the relevance judgments.
2. A larger and harder test collection: more needs, unanswerable questions that are near misses of answerable ones, and questions that need several filing years.
3. Fix the possessive-name defect in the entity matcher (a transposition-aware edit distance would also catch swapped-letter tickers such as MSTF) and re-run the affected needs.
4. Test the cross-encoder score as an abstention confidence, and stronger generators than the small model used here.
5. Evaluate the section prior on test-sized judged data (on the 10 dev needs it gained at most +0.019 nDCG@10) and learn its weights instead of setting them by hand.
6. Build and evaluate the page-image track on a GPU machine, starting with the five-page go/no-go check.

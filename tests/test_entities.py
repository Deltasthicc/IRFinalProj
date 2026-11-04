from trademesh.query.entities import EntityMatcher, bigrams, jaccard, levenshtein, normalize


def test_levenshtein_basics():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("tesla", "tesla") == 0
    assert levenshtein("tesla", "telsa") == 2          # a transposition is two edits
    assert levenshtein("apple", "aple") == 1


def test_levenshtein_limit_early_exit():
    assert levenshtein("abcdefgh", "zzzzzzzz", limit=2) == 3


def test_normalize_and_bigrams():
    assert normalize("Coca-Cola") == "cocacola"
    assert bigrams("ab") == {"$a", "ab", "b$"}
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3


def test_match_typos(companies):
    m = EntityMatcher(companies)
    assert m.match_string("Aple").ticker == "AAPL"
    assert m.match_string("Micorsoft").ticker == "MSFT"
    assert m.match_string("Tesl").ticker == "TSLA"
    assert m.match_string("TSLA").ticker == "TSLA"
    assert m.match_string("Banana Corp") is None


def test_short_ticker_one_edit_only(companies):
    m = EntityMatcher(companies)
    assert m.match_string("TSLQ").ticker == "TSLA"      # 1 edit
    assert m.match_string("TQLQ") is None               # 2 edits on a 4-letter string


def test_find_mentions_in_query(companies):
    m = EntityMatcher(companies, known_words={"apply", "risk", "compare"})
    ms = m.find_mentions("Compare Aple and Telsa supply risk")
    assert [x.ticker for x in ms] == ["AAPL", "TSLA"]
    assert [x.match.distance > 0 for x in ms] == [True, True]
    assert m.strip_mentions("Compare Aple and Telsa supply risk", ms) == "Compare and supply risk"


def test_known_word_is_not_corrected(companies):
    m = EntityMatcher(companies, known_words={"apply"})
    assert m.find_mentions("how rules apply to us") == []


def test_short_tickers_need_capitals(companies):
    comps = companies + [{"ticker": "KO", "name": "Coca-Cola Company", "aliases": ["Coke"]}]
    m = EntityMatcher(comps)
    assert m.detect_tickers("sales of KO products") == ["KO"]
    assert m.detect_tickers("ko is a word here") == []


def test_exact_name_with_legal_suffix_stripped(companies):
    m = EntityMatcher(companies)
    assert m.detect_tickers("What does Microsoft say about AI?") == ["MSFT"]
    assert m.corrections(m.find_mentions("What does Microsoft say about AI?")) == []

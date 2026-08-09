"""Language detection, sentiment, keywords, entities, topics, ranking, trends, events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pytest

from app.core.utils import utcnow
from app.database.models.article import SentimentLabel
from app.database.models.taxonomy import EntityType, TrendDirection, TrendSubject
from app.intelligence.entities import (
    RuleBasedEntityRecognizer,
    extract_entities,
    normalize_entity_name,
)
from app.intelligence.events import cluster_articles, detect_breaking
from app.intelligence.keywords import extract_keywords, keyword_strings, top_terms
from app.intelligence.language import detect_language, is_supported
from app.intelligence.pipeline import NLPPipeline, readability_score
from app.intelligence.ranking import (
    InterestProfile,
    RelevanceScorer,
    RelevanceWeights,
    recency_score,
)
from app.intelligence.sentiment import (
    LexiconSentimentAnalyzer,
    analyze_sentiment,
    score_to_label,
)
from app.intelligence.summarize import lede, split_sentences, summarize
from app.intelligence.topics import TopicClassifier, TopicDefinition, classify_topics
from app.intelligence.trends import SubjectObservation, detect_trends, window_bounds

pytestmark = pytest.mark.unit


class TestLanguageDetection:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "The government announced that the new policy will take effect from the "
                "beginning of next year and that it has been widely welcomed.",
                "en",
            ),
            (
                "El gobierno anunció que la nueva política entrará en vigor desde el "
                "próximo año y que ha sido muy bien recibida por los ciudadanos.",
                "es",
            ),
            (
                "Le gouvernement a annoncé que la nouvelle politique entrera en vigueur "
                "dès le début de l'année prochaine pour tous les citoyens.",
                "fr",
            ),
            (
                "Die Regierung hat angekündigt, dass die neue Politik ab dem nächsten "
                "Jahr in Kraft treten wird und dass sie begrüßt worden ist.",
                "de",
            ),
        ],
    )
    def test_latin_languages(self, text: str, expected: str) -> None:
        assert detect_language(text).language == expected

    def test_arabic_script_languages(self) -> None:
        arabic = detect_language("أعلنت الحكومة أن السياسة الجديدة ستدخل حيز التنفيذ العام المقبل")
        persian = detect_language("دولت اعلام کرد که سیاست جدید از سال آینده اجرایی خواهد شد")
        assert arabic.language == "ar"
        assert persian.language == "fa"

    def test_cjk_and_cyrillic(self) -> None:
        assert detect_language("政府宣布新政策将于明年年初生效并获得广泛欢迎").language == "zh"
        assert (
            detect_language("Правительство объявило о новой политике на следующий год").language
            == "ru"
        )

    def test_short_text_is_undecidable(self) -> None:
        result = detect_language("Hi")
        assert result.language is None
        assert result.confidence == 0.0

    def test_empty_input(self) -> None:
        assert detect_language(None).language is None

    def test_is_supported_gives_unknown_the_benefit_of_the_doubt(self) -> None:
        assert is_supported(None, frozenset({"en"})) is True
        assert is_supported("de", frozenset({"en"})) is False


class TestSentiment:
    def test_positive_article(self) -> None:
        result = analyze_sentiment(
            "Company reports record profits and strong growth. Analysts praised the "
            "excellent results, and shares surged to an all-time high."
        )
        assert result.score > 0.2
        assert result.label in (SentimentLabel.POSITIVE, SentimentLabel.VERY_POSITIVE)
        assert result.confidence > 0

    def test_negative_article(self) -> None:
        result = analyze_sentiment(
            "The disaster killed dozens and caused catastrophic damage. Officials warned "
            "of a worsening crisis as the death toll continued to rise."
        )
        assert result.score < -0.2
        assert result.label in (SentimentLabel.NEGATIVE, SentimentLabel.VERY_NEGATIVE)

    def test_neutral_text(self) -> None:
        result = analyze_sentiment("The meeting is scheduled for Tuesday in the main hall.")
        assert result.label is SentimentLabel.NEUTRAL

    def test_negation_flips_polarity(self) -> None:
        """This is what separates a valence model from keyword counting."""
        positive = analyze_sentiment("The results were excellent and the outcome was a success.")
        negated = analyze_sentiment("The results were not excellent and it was not a success.")
        assert positive.score > 0
        assert negated.score < positive.score

    def test_intensifier_amplifies(self) -> None:
        plain = analyze_sentiment("The outcome was good for the company overall.")
        strong = analyze_sentiment("The outcome was extremely good for the company overall.")
        assert abs(strong.score) >= abs(plain.score)

    def test_downtoner_dampens(self) -> None:
        plain = analyze_sentiment("The results were excellent across the board.")
        mild = analyze_sentiment("The results were slightly excellent across the board.")
        assert abs(mild.score) <= abs(plain.score)

    def test_contrast_marker_favours_second_clause(self) -> None:
        text = "The launch was successful, but the product failed catastrophically in tests."
        assert analyze_sentiment(text).score < 0

    def test_score_bounds_and_labels(self) -> None:
        for score, expected in [
            (-1.0, SentimentLabel.VERY_NEGATIVE),
            (-0.3, SentimentLabel.NEGATIVE),
            (0.0, SentimentLabel.NEUTRAL),
            (0.3, SentimentLabel.POSITIVE),
            (0.9, SentimentLabel.VERY_POSITIVE),
        ]:
            assert score_to_label(score) is expected

    def test_empty_input_is_neutral(self) -> None:
        result = analyze_sentiment("")
        assert result.score == 0.0
        assert result.confidence == 0.0

    def test_custom_lexicon_is_pluggable(self) -> None:
        analyzer = LexiconSentimentAnalyzer(lexicon={"frobnicate": 1.0})
        assert analyzer.analyze("They frobnicate the widget.").score > 0


class TestKeywords:
    def test_multiword_phrases_are_extracted(self) -> None:
        text = (
            "Artificial intelligence research is accelerating. Machine learning models "
            "now outperform earlier systems on artificial intelligence benchmarks."
        )
        keywords = keyword_strings(extract_keywords(text, title="Artificial intelligence advances"))
        assert any("artificial intelligence" in keyword for keyword in keywords)

    def test_title_terms_are_weighted_higher(self) -> None:
        body = "The council discussed several routine administrative matters at length."
        with_title = keyword_strings(extract_keywords(body, title="Budget approval"))
        assert any("budget" in keyword for keyword in with_title)

    def test_stopwords_and_news_noise_excluded(self) -> None:
        keywords = keyword_strings(
            extract_keywords("The spokesperson said on Monday that the report said more.")
        )
        assert not any(keyword in {"said", "monday", "the"} for keyword in keywords)

    def test_empty_input(self) -> None:
        assert extract_keywords(None) == []
        assert extract_keywords("") == []

    def test_limit_is_respected(self) -> None:
        text = " ".join(f"unique term number {index} distinct" for index in range(60))
        assert len(extract_keywords(text, limit=5)) <= 5

    def test_top_terms_over_corpus(self) -> None:
        terms = dict(top_terms(["bitcoin rally continues", "bitcoin price analysis"], limit=5))
        assert terms.get("bitcoin") == 2


class TestEntities:
    TEXT = (
        "Apple announced the results in London. Tim Cook said the company would expand. "
        "Microsoft and Google both responded. The United States government commented too."
    )

    def test_organisations_people_and_places(self) -> None:
        entities = {entity.name: entity.entity_type for entity in extract_entities(self.TEXT)}
        assert entities.get("Apple") is EntityType.ORGANIZATION
        assert entities.get("Microsoft") is EntityType.ORGANIZATION
        assert entities.get("London") is EntityType.LOCATION
        assert entities.get("United States") is EntityType.COUNTRY
        assert entities.get("Tim Cook") is EntityType.PERSON

    def test_title_mentions_count_double(self) -> None:
        without = extract_entities(self.TEXT)
        with_title = extract_entities(self.TEXT, title="Apple results")
        apple_without = next(e for e in without if e.name == "Apple")
        apple_with = next(e for e in with_title if e.name == "Apple")
        assert apple_with.mentions > apple_without.mentions

    def test_sentence_boundaries_are_not_merged(self) -> None:
        entities = {
            entity.name for entity in extract_entities("It happened Monday. Google replied.")
        }
        assert "Monday. Google" not in entities
        assert "Google" in entities

    def test_sentence_initial_common_nouns_are_ignored(self) -> None:
        names = {entity.name for entity in extract_entities("Shares rose today. Analysts agreed.")}
        assert "Shares" not in names
        assert "Analysts" not in names

    def test_salience_sums_to_one(self) -> None:
        entities = extract_entities(self.TEXT)
        assert abs(sum(entity.salience for entity in entities) - 1.0) < 0.05

    def test_normalisation_merges_variants(self) -> None:
        assert normalize_entity_name("Acme Inc.") == normalize_entity_name("ACME")

    def test_empty_text(self) -> None:
        assert extract_entities("") == []

    def test_recogniser_is_pluggable(self) -> None:
        recognizer = RuleBasedEntityRecognizer(max_entities=1)
        assert len(recognizer.extract(self.TEXT)) <= 1


class TestTopics:
    def test_classification_picks_the_right_topic(self) -> None:
        matches = classify_topics(
            "Researchers trained a large language model using deep learning techniques "
            "to improve machine translation quality.",
            title="New AI model released",
        )
        assert matches
        assert matches[0].slug == "ai"

    def test_cybersecurity_classification(self) -> None:
        matches = classify_topics(
            "A ransomware attack exploited a zero-day vulnerability, leaking credentials.",
            title="Major data breach disclosed",
        )
        assert matches[0].slug == "cybersecurity"

    def test_scores_are_normalised(self) -> None:
        matches = classify_topics("Bitcoin and ethereum prices rallied on the exchange.")
        assert matches
        assert all(0.0 <= match.score <= 1.0 for match in matches)
        assert matches[0].score == 1.0

    def test_unrelated_text_yields_nothing(self) -> None:
        assert classify_topics("Lorem ipsum dolor sit amet consectetur.") == []

    def test_custom_definitions(self) -> None:
        classifier = TopicClassifier.from_definitions(
            [TopicDefinition(slug="widgets", name="Widgets", keywords=frozenset({"widget"}))]
        )
        assert classifier.primary_category("A widget factory opened.") == "widgets"

    def test_limit_respected(self) -> None:
        matches = classify_topics(
            "The technology company reported profits while investors watched the market "
            "and the government debated new legislation on artificial intelligence.",
            limit=2,
        )
        assert len(matches) <= 2


class TestSummarisation:
    TEXT = (
        "The central bank raised interest rates by half a percentage point on Thursday. "
        "The decision was widely expected by economists across the market. "
        "Inflation has remained stubbornly above the bank's official target for months. "
        "Officials signalled that further increases may be necessary later this year. "
        "Markets reacted calmly to the announcement in afternoon trading sessions."
    )

    def test_summary_is_shorter_and_ordered(self) -> None:
        summary = summarize(self.TEXT, title="Central bank raises rates", max_sentences=2)
        assert 0 < len(summary.text) < len(self.TEXT)
        assert len(summary.sentences) <= 2
        # Sentences keep their original relative order.
        positions = [self.TEXT.index(sentence[:40]) for sentence in summary.sentences]
        assert positions == sorted(positions)

    def test_short_text_returned_as_is(self) -> None:
        assert summarize("Too short.").text

    def test_empty_input(self) -> None:
        assert summarize(None).text == ""

    def test_split_sentences_filters_fragments(self) -> None:
        assert split_sentences("Hi. " + "A properly long sentence for the splitter to keep. ")

    def test_lede_returns_first_sentence(self) -> None:
        assert lede(self.TEXT).startswith("The central bank raised")


class TestRelevanceRanking:
    def test_recency_decays(self) -> None:
        now = utcnow()
        assert recency_score(now, now=now) == pytest.approx(1.0, abs=0.01)
        assert recency_score(now - timedelta(hours=18), now=now) == pytest.approx(0.5, abs=0.02)
        assert recency_score(now - timedelta(days=7), now=now) < 0.01
        assert recency_score(None) == 0.0

    def test_fresh_beats_stale_all_else_equal(self) -> None:
        scorer = RelevanceScorer()
        fresh = scorer.score(published_at=utcnow(), quality_score=0.8)
        stale = scorer.score(published_at=utcnow() - timedelta(days=3), quality_score=0.8)
        assert fresh.score > stale.score

    def test_profile_match_raises_score(self) -> None:
        scorer = RelevanceScorer()
        profile = InterestProfile(topics=frozenset({"ai"}), keywords=frozenset({"openai"}))
        matched = scorer.score(
            published_at=utcnow(), topics=["ai"], keywords=["openai model"], profile=profile
        )
        unmatched = scorer.score(
            published_at=utcnow(), topics=["sports"], keywords=["football"], profile=profile
        )
        assert matched.score > unmatched.score

    def test_excluded_source_scores_zero(self) -> None:
        profile = InterestProfile(excluded_sources=frozenset({"tabloid"}))
        result = RelevanceScorer().score(
            published_at=utcnow(), source_slug="tabloid", profile=profile
        )
        assert result.score == 0.0

    def test_corroboration_increases_score(self) -> None:
        scorer = RelevanceScorer()
        single = scorer.score(published_at=utcnow(), corroborating_sources=1)
        many = scorer.score(published_at=utcnow(), corroborating_sources=8)
        assert many.score > single.score

    def test_score_is_bounded_and_explained(self) -> None:
        result = RelevanceScorer().score(published_at=utcnow(), quality_score=1.0)
        assert 0.0 <= result.score <= 1.0
        assert "recency" in result.components
        assert "score=" in result.explain()

    def test_weights_are_configurable(self) -> None:
        recency_only = RelevanceScorer(
            RelevanceWeights(
                source=0,
                recency=1,
                topic=0,
                keyword=0,
                entity=0,
                quality=0,
                corroboration=0,
                engagement=0,
            )
        )
        result = recency_only.score(published_at=utcnow(), quality_score=0.0)
        assert result.score == pytest.approx(1.0, abs=0.01)


class TestTrendDetection:
    @staticmethod
    def observations(count: int, key: str, sources: int = 3) -> list[SubjectObservation]:
        return [
            SubjectObservation(
                subject_type=TrendSubject.TOPIC,
                key=key,
                label=key.upper(),
                source_slug=f"source-{index % sources}",
                sentiment=0.1,
                article_id=index,
            )
            for index in range(count)
        ]

    def test_spike_is_detected(self) -> None:
        results = detect_trends(self.observations(30, "ai"), self.observations(5, "ai"))
        assert results
        trend = results[0]
        assert trend.subject_key == "ai"
        assert trend.direction is TrendDirection.RISING
        assert trend.growth_percent == pytest.approx(500.0)
        assert trend.trend_score > 0.4

    def test_new_subject_marked_new(self) -> None:
        results = detect_trends(self.observations(10, "quantum"), [])
        assert results[0].direction is TrendDirection.NEW

    def test_declining_subject(self) -> None:
        results = detect_trends(self.observations(4, "crypto"), self.observations(40, "crypto"))
        assert results[0].direction is TrendDirection.FALLING

    def test_min_articles_filter(self) -> None:
        assert detect_trends(self.observations(2, "tiny"), [], min_articles=3) == []

    def test_single_source_lowers_confidence(self) -> None:
        broad = detect_trends(self.observations(12, "ai", sources=6), [])[0]
        narrow = detect_trends(self.observations(12, "ai", sources=1), [])[0]
        assert broad.confidence > narrow.confidence
        assert broad.trend_score > narrow.trend_score

    def test_breaking_requires_growth_volume_and_breadth(self) -> None:
        breaking = detect_trends(
            self.observations(40, "ai", sources=6), self.observations(4, "ai")
        )[0]
        assert breaking.is_breaking
        quiet = detect_trends(self.observations(4, "ai", sources=1), self.observations(4, "ai"))[0]
        assert not quiet.is_breaking

    def test_window_bounds_are_adjacent(self) -> None:
        now = utcnow()
        (current_start, current_end), (previous_start, previous_end) = window_bounds(
            hours=24, now=now
        )
        assert current_end == now
        assert previous_end == current_start
        assert current_end - current_start == previous_end - previous_start


@dataclass
class FakeClusterArticle:
    id: int
    title: str
    description: str | None
    content: str | None
    source_name: str
    published_at: datetime
    sentiment_score: float = 0.0
    keywords: list[str] = field(default_factory=list)


class TestEventClustering:
    @staticmethod
    def article(article_id: int, title: str, body: str, source: str, offset: int = 0):
        return FakeClusterArticle(
            id=article_id,
            title=title,
            description=body[:100],
            content=body,
            source_name=source,
            published_at=utcnow() - timedelta(minutes=offset),
            sentiment_score=-0.2,
            keywords=["earthquake", "rescue"],
        )

    def test_same_story_across_sources_forms_one_cluster(self) -> None:
        body = (
            "A powerful earthquake struck the coastal region early on Tuesday, collapsing "
            "buildings and prompting a large rescue operation involving hundreds of workers."
        )
        articles = [
            self.article(1, "Earthquake strikes coastal region", body, "Reuters", 30),
            self.article(
                2, "Powerful earthquake hits the coast", body + " Aid arrived.", "BBC", 20
            ),
            self.article(
                3, "Rescue effort after coastal earthquake", body + " Teams dig.", "CNN", 10
            ),
        ]
        clusters = cluster_articles(articles, threshold=0.4)
        assert len(clusters) == 1
        cluster = clusters[0]
        assert cluster.size == 3
        assert cluster.source_count == 3
        assert cluster.importance > 0
        assert cluster.first_seen is not None
        assert cluster.key

    def test_unrelated_stories_stay_separate(self) -> None:
        articles = [
            self.article(
                1,
                "Earthquake strikes coastal region",
                "A powerful earthquake struck the coast, collapsing buildings and homes.",
                "Reuters",
            ),
            self.article(
                2,
                "Football club signs new striker",
                "The club completed the transfer of a striker from a rival team today.",
                "BBC",
            ),
        ]
        assert cluster_articles(articles, threshold=0.5) == []

    def test_single_source_repeats_are_not_events(self) -> None:
        body = "The company published its quarterly results showing revenue growth this year."
        articles = [
            self.article(index, f"Results update {index}", body, "SameOutlet", index)
            for index in range(3)
        ]
        assert cluster_articles(articles, require_distinct_sources=True) == []

    def test_too_few_articles(self) -> None:
        assert cluster_articles([]) == []

    def test_breaking_detection(self) -> None:
        body = "Officials confirmed the incident and emergency teams responded within minutes."
        articles = [
            self.article(index, "Incident confirmed by officials", body, f"Outlet{index}", 5)
            for index in range(3)
        ]
        clusters = cluster_articles(articles, threshold=0.3)
        assert detect_breaking(clusters, window_hours=6)


class TestNLPPipeline:
    def test_full_enrichment(self) -> None:
        result = NLPPipeline().enrich(
            title="OpenAI unveils breakthrough model as Microsoft invests billions",
            content=(
                "OpenAI announced a major breakthrough in artificial intelligence on Monday. "
                "Microsoft, which has invested billions in the startup, welcomed the launch. "
                "Critics warned about security risks and called for stronger oversight."
            ),
        )
        assert result.language == "en"
        assert result.topics and result.topics[0].slug in {"ai", "technology"}
        assert result.keywords
        assert any(entity.name == "Microsoft" for entity in result.entities)
        assert -1.0 <= result.sentiment_score <= 1.0
        assert result.word_count > 0
        assert result.errors == ()
        assert "sentiment" in result.metadata()

    def test_stage_failure_is_isolated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One broken engine must not cost the article its other enrichment."""
        import app.intelligence.pipeline as pipeline_module

        def explode(*args: object, **kwargs: object) -> None:
            raise RuntimeError("model unavailable")

        monkeypatch.setattr(pipeline_module, "extract_entities", explode)
        result = NLPPipeline().enrich(
            title="Company reports excellent results and record profits",
            content="The company reported excellent results with record profits this quarter.",
        )
        assert "entities" in result.errors
        assert result.entities == ()
        assert result.sentiment_score != 0.0

    def test_readability_bounds(self) -> None:
        score = readability_score("The cat sat on the mat. " * 20)
        assert score is None or 0.0 <= score <= 1.0
        assert readability_score("short") is None

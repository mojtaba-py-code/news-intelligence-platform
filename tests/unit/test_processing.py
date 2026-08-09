"""Cleaning, normalisation, date parsing and data-quality validation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.core.utils import utcnow
from app.processing.cleaning.html_clean import (
    extract_main_text,
    extract_meta,
    normalize_whitespace,
    sanitize_fragment,
    strip_html,
)
from app.processing.normalization.dates import parse_datetime
from app.processing.normalization.normalizer import ArticleNormalizer, NormalizationError
from app.processing.normalization.text import (
    author_key,
    normalize_author,
    normalize_category,
    normalize_country,
    normalize_language,
    normalize_title,
    word_count,
)
from app.processing.validation.quality import (
    QualityIssue,
    QualityReportBuilder,
    validate_article,
)
from app.schemas.article import RawArticle

pytestmark = pytest.mark.unit


class TestHTMLCleaning:
    def test_strip_html_removes_markup_and_entities(self) -> None:
        assert strip_html("<p>Hello <b>world</b> &amp; friends</p>") == "Hello world & friends"

    def test_script_and_style_content_is_dropped(self) -> None:
        markup = "<div>Keep<script>alert('xss')</script><style>.a{color:red}</style> this</div>"
        result = strip_html(markup)
        assert "alert" not in result
        assert "color" not in result
        assert "Keep" in result

    def test_sanitize_fragment_strips_event_handlers(self) -> None:
        result = sanitize_fragment('<p onclick="steal()">text <b>bold</b></p>')
        assert "onclick" not in result
        assert "<b>bold</b>" in result

    def test_sanitize_fragment_removes_javascript_links(self) -> None:
        result = sanitize_fragment('<a href="javascript:alert(1)">click</a>')
        assert "javascript" not in result
        assert "click" in result

    def test_sanitize_fragment_keeps_safe_links_with_rel(self) -> None:
        result = sanitize_fragment('<a href="https://example.com">link</a>')
        assert 'href="https://example.com"' in result
        assert "noopener" in result

    def test_sanitize_fragment_drops_iframes_and_svg(self) -> None:
        result = sanitize_fragment('<iframe src="https://evil"></iframe><svg onload="x()"></svg>ok')
        assert "iframe" not in result
        assert "svg" not in result
        assert "ok" in result

    def test_extract_main_text_prefers_article_body(self) -> None:
        markup = """
        <html><body>
          <nav>Home About Contact</nav>
          <article>
            <p>This is the first substantial paragraph of the news story being reported.</p>
            <p>And this is a second paragraph adding further detail about the events.</p>
          </article>
          <footer>Copyright 2026 Example</footer>
        </body></html>
        """
        text = extract_main_text(markup)
        assert "first substantial paragraph" in text
        assert "Home About Contact" not in text
        assert "Copyright" not in text

    def test_extract_main_text_drops_boilerplate_classes(self) -> None:
        markup = """
        <article>
          <div class="newsletter-signup">
            <p>Subscribe to our newsletter for daily updates now</p>
          </div>
          <p>The council approved the budget after a lengthy debate lasting several hours.</p>
        </article>
        """
        text = extract_main_text(markup)
        assert "council approved" in text
        assert "Subscribe" not in text

    def test_extract_meta_reads_opengraph(self) -> None:
        markup = """
        <html lang="en"><head>
          <title>Fallback</title>
          <meta property="og:title" content="Real Headline">
          <meta property="og:description" content="A summary.">
          <meta property="article:published_time" content="2026-01-05T10:00:00Z">
          <link rel="canonical" href="https://example.com/canonical">
        </head><body></body></html>
        """
        meta = extract_meta(markup)
        assert meta["title"] == "Real Headline"
        assert meta["description"] == "A summary."
        assert meta["canonical_url"] == "https://example.com/canonical"
        assert meta["language"] == "en"

    def test_normalize_whitespace_handles_exotic_spaces(self) -> None:
        assert normalize_whitespace("a  b\t\tc\r\n\r\n\r\nd") == "a b c\n\nd"


class TestDateParsing:
    @pytest.mark.parametrize(
        "raw",
        [
            "2026-01-05T10:30:00Z",
            "2026-01-05T10:30:00+00:00",
            "Mon, 05 Jan 2026 10:30:00 GMT",
            "2026-01-05 10:30:00",
            "2026-01-05",
            "January 5, 2026",
        ],
    )
    def test_common_formats(self, raw: str) -> None:
        parsed = parse_datetime(raw)
        assert parsed is not None
        assert parsed.tzinfo is not None
        assert parsed.year == 2026

    def test_epoch_seconds_and_milliseconds(self) -> None:
        seconds = parse_datetime(1767609000)
        millis = parse_datetime(1767609000000)
        assert seconds is not None and millis is not None
        assert seconds == millis

    def test_relative_dates(self) -> None:
        parsed = parse_datetime("3 hours ago")
        assert parsed is not None
        assert timedelta(hours=2, minutes=50) < (utcnow() - parsed) < timedelta(hours=3, minutes=10)

    def test_unparseable_returns_default(self) -> None:
        assert parse_datetime("not a date") is None
        sentinel = datetime(2020, 1, 1, tzinfo=UTC)
        assert parse_datetime("garbage", default=sentinel) == sentinel

    def test_implausible_dates_rejected(self) -> None:
        assert parse_datetime("1901-01-01") is None
        assert parse_datetime("2199-01-01") is None

    def test_naive_datetime_gets_utc(self) -> None:
        parsed = parse_datetime(datetime(2026, 1, 5, 10, 0))
        assert parsed is not None and parsed.tzinfo is UTC


class TestTextNormalisation:
    def test_title_strips_markup_and_brand_suffix(self) -> None:
        title = normalize_title(
            "<h1>Major policy shift announced by the government</h1> | BBC News"
        )
        assert title == "Major policy shift announced by the government"

    def test_short_title_keeps_suffix(self) -> None:
        # Stripping would leave too little; the whole title is kept instead.
        assert "Reuters" in normalize_title("Deal done - Reuters")

    def test_smart_quotes_normalised(self) -> None:
        assert normalize_title("The “big” announcement") == 'The "big" announcement'

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("By Jane Doe", "Jane Doe"),
            ("Jane Doe and John Roe", "Jane Doe"),
            ("By Jane Doe, CNN Staff", "Jane Doe"),
            ("newsroom@example.com", None),
            ("", None),
            ("Reuters Staff", None),
        ],
    )
    def test_author_normalisation(self, raw: str, expected: str | None) -> None:
        assert normalize_author(raw) == expected

    def test_author_key_is_accent_insensitive(self) -> None:
        assert author_key("José Álvarez") == author_key("Jose Alvarez")

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("en-US", "en"), ("English", "en"), ("fa", "fa"), ("farsi", "fa"), ("zzz", None)],
    )
    def test_language_normalisation(self, raw: str, expected: str | None) -> None:
        assert normalize_language(raw) == expected

    def test_country_normalisation(self) -> None:
        assert normalize_country("us") == "US"
        assert normalize_country("en-GB") == "GB"
        assert normalize_country("nonsense") is None

    def test_category_aliases(self) -> None:
        assert normalize_category("Tech") == "technology"
        assert normalize_category("Cyber Security") == "cybersecurity"
        assert normalize_category("crypto") == "cryptocurrency"

    def test_word_count_handles_unicode(self) -> None:
        assert word_count("سلام دنیا زیبا") == 3
        assert word_count(None) == 0


class TestNormalizer:
    def _raw(self, **overrides: object) -> RawArticle:
        payload = {
            "source_slug": "example",
            "source_name": "Example",
            "title": "  Scientists report a genuine breakthrough in battery technology  ",
            "url": "https://www.example.com/news/story?utm_source=twitter",
            "description": "<p>Researchers announced a new battery chemistry.</p>",
            "content": "<article><p>"
            + ("Detail sentence about the battery. " * 20)
            + "</p></article>",
            "author": "By Jane Doe",
            "published_at": "2026-01-05T10:00:00Z",
            "language": "en",
            "category": "Tech",
        }
        payload.update(overrides)
        return RawArticle.model_validate(payload)

    def test_normalisation_produces_canonical_record(self) -> None:
        result = ArticleNormalizer().normalize(self._raw())
        assert result.title.startswith("Scientists report")
        assert result.canonical_url == "https://example.com/news/story"
        assert "<p>" not in (result.content or "")
        assert result.author_name == "Jane Doe"
        assert result.category == "technology"
        assert len(result.content_hash) == 64
        assert len(result.title_hash) == 64
        assert result.simhash is not None
        assert 0.0 <= result.quality_score <= 1.0

    def test_identical_content_yields_identical_hash(self) -> None:
        normalizer = ArticleNormalizer()
        first = normalizer.normalize(self._raw())
        second = normalizer.normalize(
            self._raw(url="https://example.com/news/story?utm_campaign=other")
        )
        assert first.content_hash == second.content_hash

    def test_missing_title_is_rejected(self) -> None:
        with pytest.raises(NormalizationError):
            ArticleNormalizer().normalize(self._raw(title="<b> </b>"))

    def test_missing_date_falls_back_to_now(self) -> None:
        result = ArticleNormalizer().normalize(self._raw(published_at=None))
        assert (utcnow() - result.published_at).total_seconds() < 5

    def test_future_date_is_clamped(self) -> None:
        result = ArticleNormalizer().normalize(
            self._raw(published_at=utcnow() + timedelta(days=400))
        )
        assert result.published_at <= utcnow()

    def test_relative_image_url_dropped(self) -> None:
        result = ArticleNormalizer().normalize(self._raw(image_url="/img/pic.jpg"))
        assert result.image_url is None

    def test_control_characters_stripped_from_input(self) -> None:
        raw = self._raw(title="Battery\x00 breakthrough announced by researchers today")
        result = ArticleNormalizer().normalize(raw)
        assert "\x00" not in result.title


class TestQualityValidation:
    def _good(self) -> RawArticle:
        return RawArticle(
            source_slug="s",
            source_name="S",
            title="A perfectly reasonable headline about something",
            url="https://example.com/a",
            description="A description with enough words to be useful for the reader here.",
            content="Body text. " * 40,
            author="Jane Doe",
            image_url="https://example.com/a.jpg",
            published_at=utcnow(),
        )

    def test_good_article_is_valid(self) -> None:
        outcome = validate_article(self._good())
        assert outcome.is_valid
        assert outcome.score > 0.8

    def test_missing_title_is_fatal(self) -> None:
        outcome = validate_article(self._good().model_copy(update={"title": ""}))
        assert not outcome.is_valid
        assert QualityIssue.MISSING_TITLE in outcome.issues

    def test_spam_is_rejected(self) -> None:
        outcome = validate_article(
            self._good().model_copy(update={"title": "CLICK HERE to earn $5000 now buy now"})
        )
        assert not outcome.is_valid
        assert QualityIssue.SPAM_LIKE in outcome.issues

    def test_thin_content_is_a_soft_penalty(self) -> None:
        outcome = validate_article(
            self._good().model_copy(update={"content": "Two words", "description": None})
        )
        assert outcome.is_valid
        assert outcome.score < 1.0

    def test_unsupported_language_rejected_when_configured(self) -> None:
        article = self._good().model_copy(update={"language": "de"})
        outcome = validate_article(article, allowed_languages=frozenset({"en"}))
        assert not outcome.is_valid
        assert QualityIssue.UNSUPPORTED_LANGUAGE in outcome.issues

    def test_report_builder_aggregates(self) -> None:
        builder = QualityReportBuilder()
        builder.record(validate_article(self._good()))
        builder.record(validate_article(self._good().model_copy(update={"title": ""})))
        builder.record_duplicate()
        report = builder.build()
        assert report.total_processed == 2
        assert report.valid_articles == 1
        assert report.invalid_articles == 1
        assert report.missing_title == 1
        assert report.duplicate_articles == 1
        assert report.duplicate_rate == 0.5

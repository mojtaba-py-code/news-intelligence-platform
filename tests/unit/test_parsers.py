"""Feed, JSON and HTML parsers, plus robots.txt handling."""

from __future__ import annotations

import pytest

from app.core.errors import ParseError
from app.ingestion.fetchers.robots import RobotsCache, parse_robots
from app.ingestion.parsers.html import ExtractionRules, HTMLArticleParser
from app.ingestion.parsers.json_feed import (
    JSONFieldMapping,
    coerce_text,
    first_value,
    get_path,
    parse_json_items,
)
from app.ingestion.parsers.rss import parse_feed

pytestmark = pytest.mark.unit

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/"
     xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <title>Example News</title>
    <link>https://example.com</link>
    <description>Example feed</description>
    <language>en</language>
    <item>
      <title>First headline about the economy</title>
      <link>https://example.com/first</link>
      <description>A short summary of the first story.</description>
      <content:encoded><![CDATA[<p>Full body of the first story.</p>]]></content:encoded>
      <dc:creator>Jane Doe</dc:creator>
      <pubDate>Mon, 05 Jan 2026 10:00:00 GMT</pubDate>
      <guid isPermaLink="false">first-001</guid>
      <category>Business</category>
      <enclosure url="https://example.com/first.jpg" type="image/jpeg" length="1024"/>
    </item>
    <item>
      <title>Second headline about technology</title>
      <link>https://example.com/second</link>
      <description>Another summary.</description>
      <pubDate>Mon, 05 Jan 2026 11:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""

ATOM = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xml:lang="en">
  <title>Atom Example</title>
  <link href="https://atom.example.com/" rel="alternate"/>
  <subtitle>Subtitle</subtitle>
  <entry>
    <title>Atom entry headline</title>
    <link href="https://atom.example.com/entry-1" rel="alternate"/>
    <link href="https://atom.example.com/entry-1/edit" rel="edit"/>
    <id>urn:uuid:1</id>
    <summary>Entry summary text.</summary>
    <content type="html">&lt;p&gt;Entry body.&lt;/p&gt;</content>
    <author><name>John Roe</name></author>
    <published>2026-01-05T09:00:00Z</published>
    <updated>2026-01-05T09:30:00Z</updated>
    <category term="science"/>
  </entry>
</feed>
"""

#: The classic XXE payload - the parser must refuse rather than read the file.
XXE = """<?xml version="1.0"?>
<!DOCTYPE rss [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
<rss version="2.0"><channel><title>&xxe;</title>
<item><title>x</title><link>https://e.com/x</link></item></channel></rss>
"""

#: Billion laughs - exponential entity expansion.
BILLION_LAUGHS = """<?xml version="1.0"?>
<!DOCTYPE lolz [
 <!ENTITY lol "lol">
 <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
 <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
]>
<rss version="2.0"><channel><title>&lol3;</title></channel></rss>
"""


class TestRSSParsing:
    def test_rss_items_are_mapped(self) -> None:
        feed = parse_feed(RSS)
        assert feed.format == "rss"
        assert feed.title == "Example News"
        assert feed.language == "en"
        assert len(feed.items) == 2

        first = feed.items[0]
        assert first.title == "First headline about the economy"
        assert first.link == "https://example.com/first"
        assert "Full body" in first.content
        assert first.author == "Jane Doe"
        assert first.guid == "first-001"
        assert first.categories == ["Business"]
        assert first.image == "https://example.com/first.jpg"

    def test_atom_entries_are_mapped(self) -> None:
        feed = parse_feed(ATOM)
        assert feed.format == "atom"
        entry = feed.items[0]
        assert entry.title == "Atom entry headline"
        # rel="alternate" wins over rel="edit".
        assert entry.link == "https://atom.example.com/entry-1"
        assert entry.author == "John Roe"
        assert entry.categories == ["science"]

    def test_items_without_a_link_are_dropped(self) -> None:
        feed = parse_feed(
            '<rss version="2.0"><channel><item><title>No link</title></item></channel></rss>'
        )
        assert feed.items == []

    def test_xxe_is_not_expanded(self) -> None:
        """External entities must never be resolved - defusedxml refuses."""
        try:
            feed = parse_feed(XXE)
        except ParseError:
            return  # rejecting the document outright is the safest outcome
        assert "root:" not in feed.title
        assert "/etc/passwd" not in feed.title

    def test_billion_laughs_is_refused(self) -> None:
        with pytest.raises(ParseError):
            parse_feed(BILLION_LAUGHS)

    def test_malformed_xml_raises(self) -> None:
        with pytest.raises(ParseError):
            parse_feed("<rss><channel><item>")

    def test_empty_document_raises(self) -> None:
        with pytest.raises(ParseError):
            parse_feed("   ")

    def test_non_feed_document_raises(self) -> None:
        with pytest.raises(ParseError, match="Unrecognised"):
            parse_feed("<html><body>not a feed</body></html>")

    def test_oversized_feed_raises(self) -> None:
        with pytest.raises(ParseError, match="maximum"):
            parse_feed(b"x" * 11_000_000)

    def test_bytes_input_is_decoded(self) -> None:
        assert parse_feed(RSS.encode("utf-8")).items


class TestJSONMapping:
    PAYLOAD = {
        "status": "ok",
        "articles": [
            {
                "title": "API headline",
                "url": "https://api.example.com/1",
                "description": "Summary",
                "content": "Body",
                "author": "Reporter",
                "publishedAt": "2026-01-05T10:00:00Z",
                "urlToImage": "https://api.example.com/1.jpg",
                "source": {"name": "API Source"},
            },
            {"title": "", "url": "https://api.example.com/2"},
            {"title": "No URL"},
        ],
    }

    def test_default_mapping(self) -> None:
        items = parse_json_items(
            self.PAYLOAD,
            JSONFieldMapping(items_path="articles"),
            source_slug="api",
            source_name="A",
        )
        assert len(items) == 1  # the two malformed entries are skipped
        assert items[0]["title"] == "API headline"
        assert items[0]["source_name"] == "API Source"

    def test_custom_mapping_from_config(self) -> None:
        mapping = JSONFieldMapping.from_config(
            {"items_path": "data.results", "title": "headline", "url": ["link", "permalink"]}
        )
        payload = {"data": {"results": [{"headline": "H", "permalink": "https://e.com/a"}]}}
        items = parse_json_items(payload, mapping, source_slug="s", source_name="S")
        assert items[0]["title"] == "H"
        assert items[0]["url"] == "https://e.com/a"

    def test_missing_items_path_raises(self) -> None:
        with pytest.raises(ParseError):
            parse_json_items(
                {}, JSONFieldMapping(items_path="nope"), source_slug="s", source_name="S"
            )

    def test_non_list_container_raises(self) -> None:
        with pytest.raises(ParseError):
            parse_json_items(
                {"articles": "oops"},
                JSONFieldMapping(items_path="articles"),
                source_slug="s",
                source_name="S",
            )

    def test_get_path_navigates_lists_and_objects(self) -> None:
        data = {"a": {"b": [{"c": 1}]}}
        assert get_path(data, "a.b.0.c") == 1
        assert get_path(data, "a.missing") is None
        assert get_path(data, "") is data

    def test_first_value_skips_empties(self) -> None:
        assert first_value({"a": "", "b": "x"}, ("a", "b")) == "x"

    def test_coerce_text_flattens_structures(self) -> None:
        assert coerce_text({"name": "Jane"}) == "Jane"
        assert coerce_text(["a", "b"]) == "a, b"
        assert coerce_text(42) == "42"
        assert coerce_text(None) is None


HTML_PAGE = """
<html lang="en">
  <head>
    <title>Site title</title>
    <meta property="og:title" content="Real headline about the merger">
    <meta property="og:description" content="A concise standfirst.">
    <link rel="canonical" href="https://news.example.com/story-1">
    <script type="application/ld+json">
      {"@type": "NewsArticle", "headline": "LD headline",
       "datePublished": "2026-01-05T08:00:00Z",
       "author": {"name": "Jane Doe"},
       "image": {"url": "https://news.example.com/img.jpg"}}
    </script>
  </head>
  <body>
    <nav><a href="/section/business">Business</a></nav>
    <article class="story">
      <h1 class="headline">Real headline about the merger</h1>
      <div class="article-body">
        <p>The regulator approved the merger after a lengthy review process concluded.</p>
        <p>Analysts said the decision would reshape competition across the sector.</p>
      </div>
    </article>
    <div class="related"><a href="/news/other-story">Other story</a></div>
    <a href="/news/story-2">Second story</a>
    <a href="https://external.example.org/x">External</a>
  </body>
</html>
"""


class TestHTMLParser:
    def test_extraction_with_selectors(self) -> None:
        rules = ExtractionRules(title="h1.headline", content="div.article-body")
        parsed = HTMLArticleParser(rules).parse(HTML_PAGE, url="https://news.example.com/story-1")
        assert parsed["title"] == "Real headline about the merger"
        assert "regulator approved" in parsed["content"]
        assert parsed["canonical_url"] == "https://news.example.com/story-1"

    def test_falls_back_to_json_ld_and_meta(self) -> None:
        parsed = HTMLArticleParser().parse(HTML_PAGE, url="https://news.example.com/story-1")
        assert parsed["author"] == "Jane Doe"
        assert parsed["published_at"] == "2026-01-05T08:00:00Z"
        assert parsed["image_url"] == "https://news.example.com/img.jpg"
        assert parsed["description"] == "A concise standfirst."

    def test_link_extraction_resolves_relative_urls(self) -> None:
        links = HTMLArticleParser().extract_links(
            HTML_PAGE, base_url="https://news.example.com/section"
        )
        assert "https://news.example.com/news/story-2" in links
        assert all(link.startswith("http") for link in links)

    def test_link_filter_by_fragment(self) -> None:
        rules = ExtractionRules(link_must_contain=("/news/",))
        links = HTMLArticleParser(rules).extract_links(
            HTML_PAGE, base_url="https://news.example.com/"
        )
        assert links
        assert all("/news/" in link for link in links)

    def test_javascript_links_are_ignored(self) -> None:
        markup = '<a href="javascript:alert(1)">x</a><a href="https://e.com/a">y</a>'
        links = HTMLArticleParser().extract_links(markup, base_url="https://e.com")
        assert links == ["https://e.com/a"]

    def test_bad_selector_does_not_crash(self) -> None:
        rules = ExtractionRules(title="h1[[[broken")
        parsed = HTMLArticleParser(rules).parse(HTML_PAGE, url="https://news.example.com/x")
        assert parsed["title"]  # fell back to metadata


ROBOTS = """
User-agent: *
Disallow: /private/
Disallow: /admin
Allow: /private/public-page
Crawl-delay: 2

User-agent: BadBot
Disallow: /
"""


class TestRobots:
    def test_rules_are_applied_with_longest_match(self) -> None:
        policy = parse_robots(ROBOTS, "NewsIntelligenceBot/1.0")
        assert policy.can_fetch("/news/story") is True
        assert policy.can_fetch("/private/secret") is False
        assert policy.can_fetch("/private/public-page") is True
        assert policy.can_fetch("/admin/panel") is False
        assert policy.crawl_delay == 2.0

    def test_agent_specific_group_wins(self) -> None:
        policy = parse_robots(ROBOTS, "BadBot/2.0")
        assert policy.can_fetch("/anything") is False

    def test_wildcards_and_anchors(self) -> None:
        policy = parse_robots("User-agent: *\nDisallow: /*.pdf$\n", "Bot/1.0")
        assert policy.can_fetch("/files/report.pdf") is False
        assert policy.can_fetch("/files/report.html") is True

    def test_comments_are_ignored(self) -> None:
        policy = parse_robots("# comment\nUser-agent: *\nDisallow: /x # trailing\n", "Bot/1.0")
        assert policy.can_fetch("/x/y") is False

    def test_cache_stores_per_origin(self) -> None:
        cache = RobotsCache(user_agent="Bot/1.0")
        assert cache.can_fetch("https://a.example/page") is None
        cache.store("https://a.example/page", ROBOTS)
        assert cache.can_fetch("https://a.example/private/x") is False
        # A different origin has no policy yet.
        assert cache.can_fetch("https://b.example/private/x") is None

    def test_unavailable_policy_allows_crawling(self) -> None:
        cache = RobotsCache(user_agent="Bot/1.0")
        cache.store("https://a.example/x", None, available=False)
        assert cache.can_fetch("https://a.example/anything") is True

    def test_robots_url_helper(self) -> None:
        assert RobotsCache.robots_url("https://a.example/deep/page?x=1") == (
            "https://a.example/robots.txt"
        )

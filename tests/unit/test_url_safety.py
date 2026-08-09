"""SSRF protection and URL canonicalisation."""

from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.core.errors import UnsafeURLError
from app.core.url_safety import (
    _is_public_ip,
    canonicalize_url,
    clear_dns_cache,
    extract_domain,
    is_safe_url,
    validate_url,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_dns() -> None:
    clear_dns_cache()


class TestSSRFGuard:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/admin",
            "http://localhost:8000/",
            "https://localhost/",
            "http://0.0.0.0/",
            "http://10.0.0.5/internal",
            "http://192.168.1.1/router",
            "http://172.16.0.1/",
            "http://169.254.169.254/latest/meta-data/",  # cloud metadata
            "http://[::1]/",
            "http://[fd00::1]/",
            "http://metadata.google.internal/",
            "http://service.internal/",
            "http://db.local/",
        ],
    )
    def test_private_and_metadata_targets_rejected(self, url: str) -> None:
        with pytest.raises(UnsafeURLError):
            validate_url(url)

    def test_ipv4_mapped_ipv6_loopback_rejected(self) -> None:
        """``::ffff:127.0.0.1`` must not slip past the IPv4 checks."""
        assert _is_public_ip("::ffff:127.0.0.1") is False
        assert _is_public_ip("::ffff:10.0.0.1") is False

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "gopher://evil.example.com/",
            "ftp://example.com/file",
            "data:text/html,<script>alert(1)</script>",
            "javascript:alert(1)",
        ],
    )
    def test_non_http_schemes_rejected(self, url: str) -> None:
        with pytest.raises(UnsafeURLError):
            validate_url(url)

    def test_credentials_in_url_rejected(self) -> None:
        with pytest.raises(UnsafeURLError, match="credentials"):
            validate_url("https://user:secret@example.com/feed")

    def test_unusual_port_rejected(self) -> None:
        with pytest.raises(UnsafeURLError, match="port"):
            validate_url("http://93.184.216.34:6379/")

    def test_control_characters_rejected(self) -> None:
        with pytest.raises(UnsafeURLError):
            validate_url("https://example.com/\nHost: evil.com")

    def test_overlong_url_rejected(self) -> None:
        with pytest.raises(UnsafeURLError):
            validate_url("https://example.com/" + "a" * 3000)

    def test_public_ip_literal_allowed(self) -> None:
        result = validate_url("https://93.184.216.34/page")
        assert result.host == "93.184.216.34"
        assert result.is_https

    def test_is_safe_url_is_boolean(self) -> None:
        assert is_safe_url("http://127.0.0.1/") is False
        assert is_safe_url("https://93.184.216.34/") is True

    def test_protection_can_be_disabled_for_local_development(self) -> None:
        config = get_settings().model_copy(update={"ssrf_protection_enabled": False})
        # No DNS resolution, no private-range check - development only.
        result = validate_url("http://localhost:8000/feed", config=config, resolve_dns=False)
        assert result.host == "localhost"

    def test_host_allowlist_enforced(self) -> None:
        config = get_settings().model_copy(
            update={"source_host_allowlist": ["example.com"], "ssrf_protection_enabled": True}
        )
        with pytest.raises(UnsafeURLError, match=r"(?i)allowlist"):
            validate_url("https://93.184.216.34/", config=config, resolve_dns=False)


class TestCanonicalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("https://WWW.Example.COM/Article/", "https://example.com/Article"),
            ("https://example.com:443/page", "https://example.com/page"),
            ("http://example.com:80/page", "http://example.com/page"),
            ("https://example.com/page#section", "https://example.com/page"),
            ("https://example.com/page?utm_source=x&id=7", "https://example.com/page?id=7"),
            ("https://example.com/page?fbclid=abc", "https://example.com/page"),
            ("https://example.com/dir/index.html", "https://example.com/dir"),
            ("https://example.com", "https://example.com/"),
            ("example.com/news", "https://example.com/news"),
        ],
    )
    def test_canonical_forms(self, raw: str, expected: str) -> None:
        assert canonicalize_url(raw) == expected

    def test_query_parameters_are_sorted(self) -> None:
        first = canonicalize_url("https://example.com/a?b=2&a=1")
        second = canonicalize_url("https://example.com/a?a=1&b=2")
        assert first == second == "https://example.com/a?a=1&b=2"

    def test_tracking_parameters_removed_but_content_kept(self) -> None:
        result = canonicalize_url(
            "https://example.com/story?id=42&utm_campaign=spring&gclid=zz&page=2"
        )
        assert "utm_campaign" not in result
        assert "gclid" not in result
        assert "id=42" in result
        assert "page=2" in result

    def test_empty_and_invalid_input(self) -> None:
        assert canonicalize_url("") == ""
        assert canonicalize_url(None) == ""  # type: ignore[arg-type]

    def test_strip_query_option(self) -> None:
        assert canonicalize_url("https://example.com/a?x=1", strip_query=True) == (
            "https://example.com/a"
        )

    def test_canonicalisation_never_resolves_dns(self) -> None:
        # Purely syntactic, so it is safe on untrusted input.
        assert canonicalize_url("http://127.0.0.1/x") == "http://127.0.0.1/x"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("https://www.bbc.co.uk/news", "bbc.co.uk"),
            ("http://example.com", "example.com"),
            ("not a url", "not a url".lower().replace(" ", "%20").split("/")[0]),
        ],
    )
    def test_extract_domain(self, raw: str, expected: str) -> None:
        domain = extract_domain(raw)
        assert isinstance(domain, str)
        if raw.startswith("http"):
            assert domain == expected

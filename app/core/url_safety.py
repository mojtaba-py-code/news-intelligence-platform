"""URL validation, canonicalisation and SSRF protection.

The platform fetches URLs supplied by configuration files, source responses and
(indirectly) by users. Any of those can point at ``http://169.254.169.254/`` or
``http://localhost:6379/`` and turn the fetcher into a confused deputy.

Defences implemented here:

* scheme allowlist (``http``/``https`` only - no ``file://``, ``gopher://``…);
* hostname sanity checks, IDN/punycode normalisation, no credentials in URL;
* **DNS resolution followed by IP classification** - every resolved address must
  be global unicast. Loopback, private, link-local, multicast, reserved and
  IPv4-mapped-IPv6 ranges are rejected;
* an optional explicit host allowlist for locked-down deployments;
* redirect targets are re-validated (see :mod:`app.ingestion.fetchers.http`).

Note the classic TOCTOU caveat: between our resolution and the socket connect,
DNS may change (DNS rebinding). The fetcher therefore pins the validated IP as
the connection target where possible and always re-validates redirects.
"""

from __future__ import annotations

import ipaddress
import socket
from contextlib import suppress
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Final
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit

from app.core.config import Settings, get_settings
from app.core.errors import UnsafeURLError
from app.core.logging import get_logger

logger = get_logger(__name__)

MAX_URL_LENGTH: Final[int] = 2048
MAX_HOSTNAME_LENGTH: Final[int] = 253

#: Ports we are willing to talk to. Everything else (Redis 6379, Postgres 5432,
#: SMTP 25, …) is refused even on a public IP.
ALLOWED_PORTS: Final[frozenset[int]] = frozenset({80, 443, 8080, 8443})

#: Hostnames that must never resolve, regardless of DNS answers.
BLOCKED_HOSTNAMES: Final[frozenset[str]] = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
        "metadata",
        "metadata.google.internal",
        "instance-data",
        "169.254.169.254",
    }
)

#: Suffixes that only exist inside private networks.
BLOCKED_SUFFIXES: Final[tuple[str, ...]] = (
    ".local",
    ".localhost",
    ".internal",
    ".intranet",
    ".corp",
    ".home",
    ".lan",
    ".test",
    ".invalid",
    ".onion",
)

#: Tracking parameters stripped during canonicalisation (they create
#: false-negative duplicates: same article, different query string).
TRACKING_PARAMS: Final[frozenset[str]] = frozenset(
    {
        "utm_source",
        "utm_medium",
        "utm_campaign",
        "utm_term",
        "utm_content",
        "utm_id",
        "utm_name",
        "utm_reader",
        "utm_brand",
        "utm_social",
        "utm_social-type",
        "gclid",
        "gclsrc",
        "dclid",
        "fbclid",
        "msclkid",
        "mc_cid",
        "mc_eid",
        "igshid",
        "yclid",
        "_ga",
        "_gl",
        "ref",
        "ref_src",
        "ref_url",
        "referrer",
        "source",
        "cmpid",
        "campaign_id",
        "spm",
        "smid",
        "partner",
        "icid",
        "ito",
        "at_medium",
        "at_campaign",
        "sh",
        "share",
        "amp",
        "outputType",
        "CMP",
    }
)

_DEFAULT_PORTS: Final[dict[str, int]] = {"http": 80, "https": 443}


@dataclass(frozen=True, slots=True)
class URLValidationResult:
    """Outcome of :func:`validate_url`."""

    url: str
    scheme: str
    host: str
    port: int
    resolved_ips: tuple[str, ...] = field(default=())

    @property
    def is_https(self) -> bool:
        return self.scheme == "https"


def _reject(reason: str, url: str) -> UnsafeURLError:
    # The URL itself is echoed back truncated: it is attacker-influenced data.
    return UnsafeURLError(f"URL rejected: {reason}", details={"url": url[:200], "reason": reason})


def _is_public_ip(address: str) -> bool:
    """True only for globally routable unicast addresses."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False

    # Unwrap IPv4-mapped/compatible IPv6 (::ffff:127.0.0.1 must not slip through).
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        elif getattr(ip, "sixtofour", None) is not None:
            ip = ipaddress.ip_address(ip.sixtofour)  # type: ignore[arg-type]
        elif getattr(ip, "teredo", None) is not None:
            ip = ip.teredo[1]  # type: ignore[index]

    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return False
    return bool(getattr(ip, "is_global", True))


@lru_cache(maxsize=1024)
def _resolve(host: str, port: int) -> tuple[str, ...]:
    """Resolve ``host`` to every address family (cached briefly by the LRU)."""
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, UnicodeError, OSError) as exc:
        raise _reject(f"hostname could not be resolved ({exc.__class__.__name__})", host) from exc
    return tuple({str(info[4][0]) for info in infos})


def clear_dns_cache() -> None:
    """Drop memoised DNS answers (used by tests and long-running workers)."""
    _resolve.cache_clear()


def validate_url(
    raw_url: str,
    *,
    config: Settings | None = None,
    resolve_dns: bool | None = None,
) -> URLValidationResult:
    """Validate ``raw_url`` against the SSRF policy.

    Raises :class:`UnsafeURLError` when the URL must not be fetched.
    """
    config = config or get_settings()
    enforce = config.ssrf_protection_enabled
    if resolve_dns is None:
        resolve_dns = enforce

    if not raw_url or not isinstance(raw_url, str):
        raise _reject("empty URL", str(raw_url))
    url = raw_url.strip()
    if len(url) > MAX_URL_LENGTH:
        raise _reject(f"URL longer than {MAX_URL_LENGTH} characters", url)
    if any(ch in url for ch in ("\n", "\r", "\t", "\x00", " ")):
        raise _reject("URL contains control characters or whitespace", url)

    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in {s.lower() for s in config.allowed_url_schemes}:
        raise _reject(f"scheme '{scheme or '(none)'}' is not allowed", url)
    if parts.username or parts.password:
        raise _reject("URLs with embedded credentials are not allowed", url)

    hostname = (parts.hostname or "").strip().rstrip(".").lower()
    if not hostname:
        raise _reject("missing hostname", url)
    if len(hostname) > MAX_HOSTNAME_LENGTH:
        raise _reject("hostname too long", url)

    with suppress(UnicodeError):
        # Not an IDN-encodable name (e.g. a literal IP) - keep it as-is.
        hostname = hostname.encode("idna").decode("ascii")

    try:
        port = parts.port or _DEFAULT_PORTS.get(scheme, 0)
    except ValueError as exc:
        raise _reject("invalid port", url) from exc
    if enforce and port not in ALLOWED_PORTS:
        raise _reject(f"port {port} is not allowed", url)

    if enforce:
        if hostname in BLOCKED_HOSTNAMES or hostname.endswith(BLOCKED_SUFFIXES):
            raise _reject("hostname is blocked by policy", url)
        if config.source_host_allowlist and not _host_allowed(
            hostname, config.source_host_allowlist
        ):
            raise _reject("hostname is not in SOURCE_HOST_ALLOWLIST", url)

    resolved: tuple[str, ...] = ()
    literal_ip = _as_ip(hostname)
    if literal_ip is not None:
        if enforce and not _is_public_ip(literal_ip):
            raise _reject("IP address is not globally routable", url)
        resolved = (literal_ip,)
    elif resolve_dns:
        resolved = _resolve(hostname, port)
        if not resolved:
            raise _reject("hostname did not resolve to any address", url)
        unsafe = [ip for ip in resolved if not _is_public_ip(ip)]
        if unsafe:
            raise _reject("hostname resolves to a non-public address", url)

    return URLValidationResult(
        url=url, scheme=scheme, host=hostname, port=port, resolved_ips=resolved
    )


def is_safe_url(raw_url: str, *, config: Settings | None = None) -> bool:
    """Boolean form of :func:`validate_url`."""
    try:
        validate_url(raw_url, config=config)
    except UnsafeURLError:
        return False
    return True


def _as_ip(host: str) -> str | None:
    """Return the normalised IP if ``host`` is an IP literal, else ``None``."""
    candidate = host[1:-1] if host.startswith("[") and host.endswith("]") else host
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        return None


def _host_allowed(hostname: str, allowlist: list[str]) -> bool:
    """Exact match or subdomain match against the configured allowlist."""
    for entry in allowlist:
        allowed = entry.strip().lower().lstrip("*.").rstrip(".")
        if not allowed:
            continue
        if hostname == allowed or hostname.endswith(f".{allowed}"):
            return True
    return False


# --------------------------------------------------------------------------- #
# Canonicalisation - the first line of defence in deduplication
# --------------------------------------------------------------------------- #
def canonicalize_url(raw_url: str, *, strip_query: bool = False) -> str:
    """Return a stable, comparable form of ``raw_url``.

    Lowercases scheme/host, drops the default port, ``www.`` prefix, tracking
    parameters, fragments and trailing slashes, and sorts the surviving query
    parameters. Purely syntactic: it never performs network access, so it is
    safe to call on untrusted input.
    """
    if not raw_url or not isinstance(raw_url, str):
        return ""

    url = raw_url.strip()
    if not url:
        return ""
    if "://" not in url:
        url = f"https://{url.lstrip('/')}"

    parts = urlsplit(url)
    scheme = (parts.scheme or "https").lower()
    if scheme not in ("http", "https"):
        return url[:MAX_URL_LENGTH]

    host = (parts.hostname or "").lower().rstrip(".")
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return url[:MAX_URL_LENGTH]

    try:
        port = parts.port
    except ValueError:
        port = None
    netloc = host if port in (None, _DEFAULT_PORTS.get(scheme)) else f"{host}:{port}"

    path = quote(unquote(parts.path or "/"), safe="/~:@!$&'()*+,;=%")
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"
    if path.endswith(("/index.html", "/index.htm", "/index.php")):
        path = path.rsplit("/", 1)[0] or "/"

    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=False)
        if key.lower() not in TRACKING_PARAMS and not key.lower().startswith("utm_")
    ]
    query = "" if strip_query else urlencode(sorted(query_pairs), doseq=True)

    return urlunsplit((scheme, netloc, path, query, ""))[:MAX_URL_LENGTH]


def extract_domain(raw_url: str) -> str:
    """Return the registrable-ish host of a URL (``www.`` stripped), or ``""``."""
    try:
        host = urlsplit(raw_url if "://" in raw_url else f"https://{raw_url}").hostname or ""
    except ValueError:
        return ""
    host = host.lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


__all__ = [
    "ALLOWED_PORTS",
    "BLOCKED_HOSTNAMES",
    "BLOCKED_SUFFIXES",
    "MAX_URL_LENGTH",
    "TRACKING_PARAMS",
    "URLValidationResult",
    "canonicalize_url",
    "clear_dns_cache",
    "extract_domain",
    "is_safe_url",
    "validate_url",
]

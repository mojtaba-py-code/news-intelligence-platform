"""robots.txt fetching, parsing and caching.

Politeness is a functional requirement here, not decoration: a scraper that
ignores ``robots.txt`` gets the platform's IP blocked and can breach a site's
terms of use. The parser understands ``User-agent``, ``Disallow``, ``Allow``
and ``Crawl-delay``, and applies the longest-match rule from the standard.

Failures are treated conservatively-but-usably: an unreachable ``robots.txt``
(network error, 5xx) means "no policy available", so crawling proceeds at the
configured default delay; an explicit ``401``/``403`` on ``robots.txt`` itself
is treated as a refusal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Final
from urllib.parse import urlsplit, urlunsplit

from app.core.logging import get_logger

logger = get_logger(__name__)

CACHE_TTL_SECONDS: Final[int] = 3600
MAX_ROBOTS_BYTES: Final[int] = 512_000
MAX_RULES: Final[int] = 2_000


@dataclass(frozen=True, slots=True)
class RobotsPolicy:
    """Parsed rules that apply to one user agent."""

    allowed: tuple[str, ...] = ()
    disallowed: tuple[str, ...] = ()
    crawl_delay: float | None = None
    fetched_at: float = 0.0
    available: bool = True

    def can_fetch(self, path: str) -> bool:
        """Longest-match wins; ``Allow`` beats ``Disallow`` on equal length."""
        if not self.available:
            return True
        target = path or "/"
        best_allow = max((len(rule) for rule in self.allowed if _matches(target, rule)), default=-1)
        best_deny = max(
            (len(rule) for rule in self.disallowed if _matches(target, rule)), default=-1
        )
        if best_deny < 0:
            return True
        return best_allow >= best_deny


def _matches(path: str, rule: str) -> bool:
    """Support the ``*`` wildcard and the ``$`` end-anchor from the standard."""
    if not rule:
        return False
    if rule == "/":
        return True
    if "*" not in rule and "$" not in rule:
        return path.startswith(rule)

    anchored = rule.endswith("$")
    pattern = rule[:-1] if anchored else rule
    segments = pattern.split("*")

    position = 0
    for index, segment in enumerate(segments):
        if not segment:
            continue
        if index == 0:
            if not path.startswith(segment):
                return False
            position = len(segment)
            continue
        found = path.find(segment, position)
        if found < 0:
            return False
        position = found + len(segment)
    if anchored:
        return path.endswith(segments[-1]) if segments[-1] else True
    return True


def parse_robots(content: str, user_agent: str) -> RobotsPolicy:
    """Extract the rules that apply to ``user_agent`` (falling back to ``*``)."""
    agent_token = user_agent.split("/")[0].strip().lower()

    groups: dict[str, dict[str, list[str]]] = {}
    delays: dict[str, float] = {}
    current_agents: list[str] = []
    expecting_agents = True

    for raw_line in content[:MAX_ROBOTS_BYTES].splitlines()[:MAX_RULES]:
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        field_name, _, value = line.partition(":")
        field_name = field_name.strip().lower()
        value = value.strip()

        if field_name == "user-agent":
            if not expecting_agents:
                current_agents = []
                expecting_agents = True
            agent = value.lower()
            current_agents.append(agent)
            groups.setdefault(agent, {"allow": [], "disallow": []})
        elif field_name in ("allow", "disallow"):
            expecting_agents = False
            for agent in current_agents or ["*"]:
                groups.setdefault(agent, {"allow": [], "disallow": []})
                if value:
                    groups[agent]["allow" if field_name == "allow" else "disallow"].append(value)
                elif field_name == "disallow":
                    # "Disallow:" with an empty value means "allow everything".
                    continue
        elif field_name == "crawl-delay":
            expecting_agents = False
            try:
                delay = float(value)
            except ValueError:
                continue
            for agent in current_agents or ["*"]:
                delays[agent] = max(0.0, min(delay, 300.0))

    selected = None
    for agent in groups:
        if agent != "*" and agent in agent_token:
            selected = agent
            break
    if selected is None:
        selected = "*" if "*" in groups else next(iter(groups), None)

    if selected is None:
        return RobotsPolicy(fetched_at=time.monotonic())

    rules = groups[selected]
    return RobotsPolicy(
        allowed=tuple(rules["allow"]),
        disallowed=tuple(rules["disallow"]),
        crawl_delay=delays.get(selected, delays.get("*")),
        fetched_at=time.monotonic(),
    )


@dataclass
class RobotsCache:
    """Per-origin robots.txt cache with a TTL."""

    user_agent: str
    ttl_seconds: int = CACHE_TTL_SECONDS
    _policies: dict[str, RobotsPolicy] = field(default_factory=dict, init=False)

    @staticmethod
    def origin_of(url: str) -> str:
        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc, "", "", ""))

    @staticmethod
    def robots_url(url: str) -> str:
        parts = urlsplit(url)
        return urlunsplit((parts.scheme, parts.netloc, "/robots.txt", "", ""))

    def get(self, url: str) -> RobotsPolicy | None:
        policy = self._policies.get(self.origin_of(url))
        if policy is None:
            return None
        if time.monotonic() - policy.fetched_at > self.ttl_seconds:
            self._policies.pop(self.origin_of(url), None)
            return None
        return policy

    def store(self, url: str, content: str | None, *, available: bool = True) -> RobotsPolicy:
        if content is None or not available:
            policy = RobotsPolicy(fetched_at=time.monotonic(), available=available)
        else:
            policy = parse_robots(content, self.user_agent)
        self._policies[self.origin_of(url)] = policy
        return policy

    def clear(self) -> None:
        self._policies.clear()

    def can_fetch(self, url: str) -> bool | None:
        """``True``/``False`` when a policy is cached, ``None`` when unknown."""
        policy = self.get(url)
        if policy is None:
            return None
        return policy.can_fetch(urlsplit(url).path or "/")

    def crawl_delay(self, url: str) -> float | None:
        policy = self.get(url)
        return policy.crawl_delay if policy else None


__all__ = ["RobotsCache", "RobotsPolicy", "parse_robots"]

"""Malformed input is refused, not repaired, in two more places.

Both are the mistake `parse_route_prefix`'s own docstring names when it refuses a bare
word: a value that has to be guessed at is a value the operator did not mean, and the
guess surfaces later as a misrouted request or a suite that skipped, rather than now as
an error.
"""

from __future__ import annotations

import pytest
from container.common.config import ConfigError, _interval, parse_route_prefix


@pytest.mark.parametrize("raw", ["/", "//", "///", " / ", " // "])
def test_a_prefix_that_normalises_to_nothing_is_refused(raw: str) -> None:
    """A value whose normalised form is empty must not read as "no prefix".

    That is the dangerous direction: the deployment believes it set a prefix, the
    container serves the routes bare, and nothing says so until a request arrives at a
    path nobody expected it to answer.
    """
    with pytest.raises(ConfigError):
        parse_route_prefix(raw)


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_an_absent_prefix_still_means_no_prefix(raw: str | None) -> None:
    """Non-vacuity for the refusal above: unset is a legitimate answer.

    A guard that refused everything would make the test above pass while breaking every
    deployment that does not use a prefix, which is all of them today.
    """
    assert parse_route_prefix(raw) == ""


@pytest.mark.parametrize(
    "raw,expected", [("/c/frontdesk", "/c/frontdesk"), ("/c/frontdesk/", "/c/frontdesk")]
)
def test_a_real_prefix_is_accepted_and_normalised(raw: str, expected: str) -> None:
    assert parse_route_prefix(raw) == expected


@pytest.mark.parametrize("raw", ["0", "-1", "-60"])
def test_a_backup_cadence_that_is_not_a_cadence_is_refused(monkeypatch, raw: str) -> None:
    """Zero or negative is a busy loop, not a faster backup.

    The wait between cycles returns immediately, so the task uploads continuously
    instead of serving turns. This is the only value the container can say is wrong, so
    it is the only one refused.
    """
    monkeypatch.setenv("SMC_BACKUP_INTERVAL_SECS", raw)
    with pytest.raises(ConfigError, match="not a cadence"):
        _interval("SMC_BACKUP_INTERVAL_SECS", 60)


@pytest.mark.parametrize("raw", ["1", "2", "5", "900"])
def test_a_short_but_real_cadence_is_accepted(monkeypatch, raw: str) -> None:
    """There is no floor above zero, and that absence is the decision.

    A floor would be a claim about what a cycle costs on a real data home, and nothing
    here has measured one. An operator running a crew with three small transcripts who
    asks for two seconds is not making a mistake this module can see.
    """
    monkeypatch.setenv("SMC_BACKUP_INTERVAL_SECS", raw)
    assert _interval("SMC_BACKUP_INTERVAL_SECS", 60) == int(raw)

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional


DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
MONTH_ONLY_RE = re.compile(r"^\d{4}-\d{2}$")
YEAR_ONLY_RE = re.compile(r"^\d{4}$")
BETWEEN_RE = re.compile(r"^between\s+(.+?)\s+and\s+(.+)$", re.IGNORECASE)
PREFIX_RE = re.compile(r"^(on|before|after|around)\s+(.+)$", re.IGNORECASE)


@dataclass
class TimeSpan:
    start: datetime
    end: datetime
    granularity: str
    raw: str = ""


@dataclass
class TimeConstraint:
    mode: str
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    granularity: str = "unknown"
    raw: str = ""


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.replace(tzinfo=None)


def _start_of_day(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 0, 0, 0)


def _end_of_day(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 23, 59, 59, 999999)


def _span_from_date(day: date, *, raw: str) -> TimeSpan:
    return TimeSpan(
        start=_start_of_day(day),
        end=_end_of_day(day),
        granularity="day",
        raw=raw,
    )


def _parse_time_span(text: str) -> Optional[TimeSpan]:
    value = _clean(text)
    if not value:
        return None

    if DATE_ONLY_RE.fullmatch(value):
        year, month, day = map(int, value.split("-"))
        try:
            return _span_from_date(date(year, month, day), raw=value)
        except ValueError:
            return None

    if MONTH_ONLY_RE.fullmatch(value):
        year, month = map(int, value.split("-"))
        last_day = calendar.monthrange(year, month)[1]
        return TimeSpan(
            start=_start_of_day(date(year, month, 1)),
            end=_end_of_day(date(year, month, last_day)),
            granularity="month",
            raw=value,
        )

    if YEAR_ONLY_RE.fullmatch(value):
        year = int(value)
        return TimeSpan(
            start=_start_of_day(date(year, 1, 1)),
            end=_end_of_day(date(year, 12, 31)),
            granularity="year",
            raw=value,
        )

    for candidate in (value, value.replace("Z", "+00:00")):
        try:
            point = datetime.fromisoformat(candidate)
            return TimeSpan(
                start=point,
                end=point,
                granularity="datetime",
                raw=value,
            )
        except Exception:
            continue
    return None


def parse_time_constraint(text: str) -> Optional[TimeConstraint]:
    value = _clean(text)
    if not value:
        return None

    between_match = BETWEEN_RE.fullmatch(value)
    if between_match:
        left = _parse_time_span(between_match.group(1))
        right = _parse_time_span(between_match.group(2))
        if left and right:
            return TimeConstraint(
                mode="between",
                start=left.start,
                end=right.end,
                granularity=max(left.granularity, right.granularity),
                raw=value,
            )
        return None

    prefix_match = PREFIX_RE.fullmatch(value)
    if prefix_match:
        mode = prefix_match.group(1).lower()
        span = _parse_time_span(prefix_match.group(2))
        if not span:
            return None
        if mode == "on":
            return TimeConstraint(
                mode="on",
                start=span.start,
                end=span.end,
                granularity=span.granularity,
                raw=value,
            )
        if mode == "before":
            return TimeConstraint(
                mode="before",
                end=span.start,
                granularity=span.granularity,
                raw=value,
            )
        if mode == "after":
            return TimeConstraint(
                mode="after",
                start=span.end,
                granularity=span.granularity,
                raw=value,
            )
        if mode == "around":
            if span.granularity == "day":
                delta = timedelta(days=3)
            elif span.granularity == "month":
                delta = timedelta(days=31)
            elif span.granularity == "year":
                delta = timedelta(days=366)
            else:
                delta = timedelta(days=1)
            return TimeConstraint(
                mode="around",
                start=span.start - delta,
                end=span.end + delta,
                granularity=span.granularity,
                raw=value,
            )
        return None

    span = _parse_time_span(value)
    if span:
        return TimeConstraint(
            mode="on",
            start=span.start,
            end=span.end,
            granularity=span.granularity,
            raw=value,
        )
    return None


def extract_record_time_spans(record: Dict[str, Any]) -> List[TimeSpan]:
    spans: List[TimeSpan] = []
    dimension = record.get("dimension") if isinstance(record.get("dimension"), dict) else {}
    raw_dimension_time = _clean(dimension.get("time"))
    if raw_dimension_time:
        span = _parse_time_span(raw_dimension_time)
        if span:
            spans.append(span)
    source_time = record.get("source_time")
    if source_time is not None:
        if isinstance(source_time, datetime):
            spans.append(
                TimeSpan(
                    start=source_time,
                    end=source_time,
                    granularity="datetime",
                    raw=source_time.isoformat(),
                )
            )
        elif isinstance(source_time, str) and source_time.strip():
            try:
                dt = datetime.fromisoformat(source_time.strip())
                spans.append(
                    TimeSpan(start=dt, end=dt, granularity="datetime", raw=source_time.strip())
                )
            except ValueError:
                pass
    return spans


def time_constraint_match_score(constraints: List[str], record: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = [_clean(value) for value in constraints if _clean(value)]
    if not cleaned:
        return {
            "score": 0.0,
            "matched": False,
            "mode": "",
            "matched_constraints": [],
            "record_time_spans": [],
        }

    record_spans = extract_record_time_spans(record)
    if not record_spans:
        return {
            "score": 0.0,
            "matched": False,
            "mode": "",
            "matched_constraints": [],
            "record_time_spans": [],
        }

    matched_constraints: List[str] = []
    modes: List[str] = []

    for raw_constraint in cleaned:
        parsed = parse_time_constraint(raw_constraint)
        if not parsed:
            continue
        modes.append(parsed.mode)
        parsed_start = _naive(parsed.start) if parsed.start is not None else None
        parsed_end = _naive(parsed.end) if parsed.end is not None else None
        matched = False
        for span in record_spans:
            span_start = _naive(span.start)
            span_end = _naive(span.end)
            if parsed.mode == "on":
                matched = span_start <= parsed_end and span_end >= parsed_start
            elif parsed.mode == "between":
                matched = span_start <= parsed_end and span_end >= parsed_start
            elif parsed.mode == "before":
                matched = span_end < parsed_end
            elif parsed.mode == "after":
                matched = span_start > parsed_start
            elif parsed.mode == "around":
                matched = span_start <= parsed_end and span_end >= parsed_start
            if matched:
                matched_constraints.append(raw_constraint)
                break

    score = len(matched_constraints) / max(1, len(cleaned))
    return {
        "score": max(0.0, min(1.0, score)),
        "matched": bool(matched_constraints),
        "mode": modes[0] if modes else "",
        "matched_constraints": matched_constraints,
        "record_time_spans": [
            {
                "start": span.start.isoformat(),
                "end": span.end.isoformat(),
                "granularity": span.granularity,
                "raw": span.raw,
            }
            for span in record_spans
        ],
    }


def record_satisfies_time_constraint(constraint_text: str, record: Dict[str, Any]) -> bool:
    result = time_constraint_match_score([constraint_text], record)
    return bool(result["matched"])


def filter_records_by_time_constraint(records: List[Dict[str, Any]], constraint_text: str) -> List[Dict[str, Any]]:
    return [record for record in records if record_satisfies_time_constraint(constraint_text, record)]

"""Timestamp formatting and (later) citation parsing.

Kept in its own module so commit #7's citation regex sits alongside the
display formatter instead of leaking into transcripts.py.
"""


def seconds_to_hhmmss(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

from datetime import datetime, timezone
from urllib.parse import urlparse

from dateutil.parser import parse as datetime_parse


def str_to_date(date_str):
    if isinstance(date_str, datetime):
        return date_str.astimezone(tz=timezone.utc)

    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%zZ"):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    return datetime_parse(date_str).replace(tzinfo=timezone.utc)


def is_valid_url(url):
    try:
        u = urlparse(url)
        return u.scheme != ""
    except ValueError:
        return False

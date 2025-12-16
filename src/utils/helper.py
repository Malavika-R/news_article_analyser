def clean_url_list(raw_input: str):
    """
    Converts text area input into a cleaned list of unique URLs.
    """
    urls = [u.strip() for u in raw_input.split("\n") if u.strip()]
    return list(set(urls))  # remove duplicates

import hashlib

def hash_urls(urls: list[str]) -> str:
    joined = "|".join(sorted(urls))
    return hashlib.md5(joined.encode()).hexdigest()

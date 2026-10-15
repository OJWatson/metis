"""Reject missing local page/download/asset links in a rendered Quarto site."""

from html.parser import HTMLParser
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attributes):
        values = dict(attributes)
        self.links.extend(values[key] for key in ("href", "src") if values.get(key))


def check(root):
    root = Path(root).resolve()
    pages = list(root.rglob("*.html"))
    if not pages:
        raise ValueError(f"No rendered HTML pages in {root}")
    missing = []
    for page in pages:
        parser = Links()
        parser.feed(page.read_text(encoding="utf-8"))
        for link in parser.links:
            url = urlsplit(link)
            if url.scheme or url.netloc or not url.path:
                continue
            # Root-relative links depend on the hosting prefix; Quarto emits
            # relative links for local site content, which are checked here.
            if url.path.startswith("/"):
                continue
            target = (page.parent / unquote(url.path)).resolve()
            if not target.exists():
                missing.append(f"{page.relative_to(root)} -> {link}")
    if missing:
        raise ValueError("Missing local targets:\n" + "\n".join(missing))
    print(f"Checked links and assets in {len(pages)} HTML pages")


if __name__ == "__main__":
    check(sys.argv[1] if len(sys.argv) > 1 else "quarto/_site")

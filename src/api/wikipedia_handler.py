import wikipediaapi
from urllib.parse import urlparse, parse_qs, unquote


def get_page_from_url(wiki_wiki: wikipediaapi.Wikipedia, url: str):
    """
    Return a wikipediaapi.Page for a given Wikipedia URL or page title.
    Accepts full URLs like:
      https://en.wikipedia.org/wiki/Python_(programming_language)
    or short paths like:
      /wiki/Python_(programming_language)
    or plain titles like:
      Python (programming language)
    Raises ValueError if the page does not exist or the input is empty.
    """

    if not url:
        raise ValueError("Empty URL or title provided")

    parsed = urlparse(url)
    if parsed.scheme or parsed.netloc or parsed.path.startswith("/"):
        title = ""
        if parsed.path.startswith("/wiki/"):
            title = parsed.path.split("/wiki/", 1)[1]
        elif parsed.path == "/w/index.php":
            title = parse_qs(parsed.query).get("title", [""])[0]
        else:
            title = parsed.path.rsplit("/", 1)[-1]
    else:
        title = url

    title = unquote(title).replace("_", " ").strip()
    page = wiki_wiki.page(title)
    if not page.exists():
        raise ValueError(f"Wikipedia page not found: {title}")
    return page

def get_plot_section(page: wikipediaapi.WikipediaPage) -> str:
    """
    Return the text of the plot (or similar) section from a wikipediaapi.Page.
    Returns empty string if none found.
    """
    target = {"plot", "plot summary", "plot synopsis", "synopsis", "premise", "summary"}

    for s in page.sections:
        title = s.title.strip().lower()
        if title in target:
            return s.text.strip()

    return None
import wikipediaapi
from urllib.parse import urlparse, parse_qs, unquote
from typing import Optional, Tuple
import re


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


def get_plot_section(page: wikipediaapi.WikipediaPage) -> tuple[Optional[str], Optional[str]]:
    """
    Return the text of the plot (or similar) section from a wikipediaapi.Page.
    Returns None if none found.
    
    Args:
        page: Wikipedia page object
        
    Returns:
        Plot text as string and wiki section title if found, else (None, None)
    """
    target = [
        "plot", "plot summary", "plot synopsis",
        "synopsis", "summary", "story", "storyline", "premise" 
    ]
    for t in target:
        for s in page.sections:
            if s.title.strip().lower() == t:
                return s.text, t

    return None, None


def clean_plot_text(plot: str) -> str:
    """
    Clean plot text for CSV storage.
    Removes line breaks and normalizes whitespace.
    
    Args:
        plot: Raw plot text with line breaks
        
    Returns:
        Cleaned plot text with line breaks replaced by spaces
    """
    if not plot:
        return ''
    
    # Convert to string if not already
    plot_str = str(plot)
    
    # Replace all types of line breaks with spaces
    plot_str = re.sub(r'\r\n|\r|\n', ' ', plot_str)
    
    # Replace multiple consecutive spaces with single space
    plot_str = re.sub(r' +', ' ', plot_str)
    
    # Strip leading/trailing whitespace
    plot_str = plot_str.strip()
    
    return plot_str


def fetch_plot_from_url(
    wikipedia_link: str,
    user_agent: str = 'GroupDataLiteracy/1.0 (movie data pipeline)'
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Fetch plot from Wikipedia for a given URL.
    This is the main function to call from the pipeline.
    
    Args:
        wikipedia_link: Wikipedia URL for the movie
        user_agent: User agent string for Wikipedia API
        
    Returns:
        Tuple of (plot_text, error_message)
        - plot_text: Retrieved plot text (or None if failed)
        - section_title: Title of the section from which plot was extracted (or None if not found)
        - error_message: Error message if failed (or None if success)
    """
    if not wikipedia_link:
        return (None, None, None)
    
    # Initialize Wikipedia API (each call gets its own instance for thread safety)
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent=user_agent,
        language='en'
    )
    
    try:
        # Get Wikipedia page
        page = get_page_from_url(wiki_wiki, wikipedia_link)
        
        # Extract plot section
        plot, section_title = get_plot_section(page)
        
        if plot:
            # Clean the plot text (remove line breaks, normalize whitespace)
            cleaned_plot = clean_plot_text(plot)
            return (cleaned_plot, section_title, None)
        return (None, None, "No plot found")
                
    except ValueError as e:
        error_msg = str(e)
        return (None, None, error_msg)
    except Exception as e:
        error_msg = str(e)
        return (None, None, error_msg)
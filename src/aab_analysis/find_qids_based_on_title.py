"""
Find QIDs for movies based on their titles.

This script takes a list of movie titles and returns their corresponding QIDs
(movie_id) if they exist in the dataset.
"""

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(BASE_DIR, "src")

# Add BASE_DIR to Python path so imports work
sys.path.insert(0, BASE_DIR)

# Import functions from data_utils
from src.utils.data_utils import (  # type: ignore  # noqa: E402
    load_final_dataset,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(BASE_DIR, "data", "data_final")
CSV_PATH = os.path.join(BASE_DIR, "data", "data_final", "final_dataset.csv")
START_YEAR = 1930
END_YEAR = 2024


def find_qid_by_title(
    title: str,
    movie_data: pd.DataFrame,
    case_sensitive: bool = False,
) -> List[str]:
    """
    Find QID(s) for a movie title in the dataset.

    Parameters:
    - title: Movie title to search for
    - movie_data: DataFrame containing movie data
    - case_sensitive: Whether to match case (default: False)

    Returns:
    - List of QIDs (movie_id) that match the title
    """
    if movie_data.empty:
        return []

    if "movie_id" not in movie_data.columns or "title" not in movie_data.columns:
        raise ValueError("DataFrame must contain 'movie_id' and 'title' columns")

    # Filter out rows with missing movie_id or title
    df_filtered = movie_data[
        movie_data["movie_id"].notna() & movie_data["title"].notna()
    ].copy()

    if df_filtered.empty:
        return []

    # Exact match (case-insensitive by default)
    title_clean = title.strip()
    if case_sensitive:
        matches = df_filtered[df_filtered["title"] == title_clean]
    else:
        matches = df_filtered[
            df_filtered["title"].str.strip().str.lower() == title_clean.lower()
        ]

    if matches.empty:
        return []

    # Return list of unique QIDs
    qids = matches["movie_id"].unique().tolist()
    return qids


def find_qids_for_titles(
    titles: List[str],
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    csv_path: str = CSV_PATH,
    case_sensitive: bool = False,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Find QIDs for a list of movie titles.

    Parameters:
    - titles: List of movie titles to search for
    - start_year: First year to filter movies (default: 1930)
    - end_year: Last year to filter movies (default: 2024)
    - csv_path: Path to final_dataset.csv
    - case_sensitive: Whether to match case (default: False)

    Returns:
    - Tuple of (flat list of all QIDs, mapping of title -> list of QIDs)
    """
    logger.info(f"{'=' * 60}")
    logger.info("Finding QIDs for movie titles")
    logger.info(f"{'=' * 60}")

    # Load movie metadata
    logger.info("Loading movie metadata...")
    movie_data = load_final_dataset(csv_path, verbose=False)

    if movie_data.empty:
        raise ValueError(f"No movie data found in {csv_path}")

    logger.info(f"Loaded {len(movie_data)} movies from metadata file")

    # Filter by year range if year column exists
    if "year" in movie_data.columns:
        movie_data = movie_data[
            (movie_data["year"] >= start_year) & (movie_data["year"] <= end_year)
        ].copy()
        logger.info(
            f"Filtered to {len(movie_data)} movies between {start_year} and {end_year}"
        )

    # Find QIDs for each title; track both flat list and mapping
    all_qids = []
    title_to_qids = {}
    for title in titles:
        qids = find_qid_by_title(title, movie_data, case_sensitive=case_sensitive)
        title_to_qids[title] = qids
        all_qids.extend(qids)
        if qids:
            logger.info(
                f"'{title}': Found {len(qids)} match(es) - QID(s): {', '.join(qids)}"
            )
            for qid in qids:
                movie_info = movie_data[movie_data["movie_id"] == qid]
                if not movie_info.empty:
                    year = movie_info.iloc[0].get("year", "N/A")
                    found_title = movie_info.iloc[0]["title"]
                    logger.info(f"  QID {qid}: '{found_title}' (Year: {year})")
        else:
            logger.warning(f"'{title}': Not found in dataset")

    return all_qids, title_to_qids


def main(
    titles: Optional[List[str]] = None,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    csv_path: str = CSV_PATH,
    case_sensitive: bool = False,
) -> List[str]:
    """
    Main function to find QIDs for movie titles.

    Parameters:
    - titles: List of movie titles to search for
    - start_year: First year to filter movies
    - end_year: Last year to filter movies
    - csv_path: Path to final_dataset.csv
    - case_sensitive: Whether to match case

    Returns:
    - List of QIDs (movie_id) that match the input titles (flat list; extra requirement)
    """
    if titles is None:
        # Example titles
        titles = [
            "A Few Good Men",
            "The Firm",
            "Philadelphia",
            "Primal Fear",
        ]
        logger.info("Using default example titles")

    qids, title_to_qids = find_qids_for_titles(
        titles=titles,
        start_year=start_year,
        end_year=end_year,
        csv_path=csv_path,
        case_sensitive=case_sensitive,
    )

    # Print summary
    logger.info(f"\n{'=' * 60}")
    logger.info("Summary:")
    logger.info(f"{'=' * 60}")
    logger.info(f"Located {len(qids)} total QIDs from {len(titles)} titles")
    logger.info("")

    # Print formatted results (show title: QIDs mapping as info)
    for title in titles:
        title_qids = title_to_qids.get(title, [])
        if title_qids:
            logger.info(f"{title}: {', '.join(title_qids)}")
        else:
            logger.info(f"{title}: NOT FOUND")

    return qids


if __name__ == "__main__":
    # Example usage
    qids = main(
        titles=[
            "Apocalypse Now",
            "Platoon",
            "Full Metal Jacket",
            "The Deer Hunter",
            "Hamburger Hill",
            "We Were Soldiers",
            "Born on the Fourth of July",
            "Casualties of War",
            "Rescue Dawn",
            "The Green Berets",
            "Tigerland",
            "Heaven & Earth",
            "84 Charlie MoPic",
            "Bat21",
            "Coming Home",
            "Go Tell the Spartans",
            "In the Year of the Pig",
            "The Boys in Company C",
            "Purple Hearts",
            "Da 5 Bloods",
        ],
    )
    # qids is a flat list of QID strings
    print(qids)

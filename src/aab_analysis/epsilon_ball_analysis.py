"""
Epsilon ball analysis for finding movies within a distance threshold around anchor movies.

This script finds all movies within an epsilon distance (epsilon ball) around
specified anchor movies. The anchor can be a single movie or the average of
multiple movies. Results include distances, rankings, and temporal analysis.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(BASE_DIR, "src")

# Add BASE_DIR to Python path so imports work
sys.path.insert(0, BASE_DIR)

# Import functions from math_functions and data_utils
from src.aab_analysis.math_functions import (  # type: ignore  # noqa: E402
    compute_anchor_embedding,
    find_movies_in_epsilon_ball,
    interpret_ks_test,
    kolmogorov_smirnov_test,
    kolmogorov_smirnov_test_temporal,
)
from src.aab_analysis.utils.epsilon_ball_utils import (  # type: ignore  # noqa: E402
    compute_embeddings_hash,
    get_anchor_names_string,
    load_cached_mean_embedding,
    save_cached_mean_embedding,
    truncate_filename_component,
)
from src.aab_analysis.visualizations.epsilon_ball_visualization import (  # type: ignore  # noqa: E402
    plot_distance_distribution,
    plot_ks_test_cdf,
    plot_ks_test_temporal_cdf,
    plot_movies_over_time,
)
from src.utils.data_utils import (  # type: ignore  # noqa: E402
    load_final_dataset,
    load_final_dense_embeddings,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(BASE_DIR, "data", "data_final")
CSV_PATH = os.path.join(BASE_DIR, "data", "data_final", "final_dataset.csv")
CACHE_DIR = os.path.join(BASE_DIR, "data", "cache", "mean_embeddings")
START_YEAR = 1930
END_YEAR = 2024


def analyze_epsilon_ball(
    anchor_qids: list,
    epsilon: float,
    filtered_embeddings: np.ndarray,
    filtered_movie_ids: np.ndarray,
    movie_data: pd.DataFrame,
    anchor_method: str = "average",
    exclude_anchors: bool = True,
):
    """
    Analyze movies within an epsilon ball around anchor movies.

    Parameters:
    - anchor_qids: List of movie QIDs to use as anchors
    - epsilon: Maximum cosine distance threshold (0 <= epsilon <= 2)
    - filtered_embeddings: Array of embeddings already filtered by year range
        (shape: [n_movies, embedding_dim])
    - filtered_movie_ids: Array of movie IDs corresponding to filtered_embeddings
    - movie_data: DataFrame with movie metadata already filtered by year range
    - anchor_method: Method to combine anchor embeddings ("average" or "medoid")
    - exclude_anchors: Whether to exclude anchor movies from results

    Returns:
    - DataFrame with columns: movie_id, title, year, distance, similarity, rank
    """
    logger.info(f"{'=' * 60}")
    logger.info("Epsilon ball analysis")
    logger.info(f"Anchor QIDs: {anchor_qids}")
    logger.info(f"Epsilon: {epsilon}")
    logger.info(f"{'=' * 60}")

    try:
        logger.info("Computing anchor embedding...")
        anchor_embedding = compute_anchor_embedding(
            anchor_qids=anchor_qids,
            embeddings_corpus=filtered_embeddings,
            movie_ids=filtered_movie_ids,
            method=anchor_method,
        )

        anchor_titles = []
        for qid in anchor_qids:
            anchor_movie = movie_data[movie_data["movie_id"] == qid]
            if not anchor_movie.empty:
                anchor_titles.append(anchor_movie.iloc[0]["title"])
            else:
                anchor_titles.append("Unknown")

        logger.info(f"Anchor movies: {list(zip(anchor_qids, anchor_titles))}")

        logger.info(f"Finding movies within epsilon ball (epsilon={epsilon})...")
        exclude_anchor_ids = anchor_qids if exclude_anchors else None
        indices, distances, similarities = find_movies_in_epsilon_ball(
            embeddings_corpus=filtered_embeddings,
            anchor_embedding=anchor_embedding,
            movie_ids=filtered_movie_ids,
            epsilon=epsilon,
            exclude_anchor_ids=exclude_anchor_ids,
        )

        logger.info(f"Found {len(indices)} movies within epsilon ball")

        if len(indices) == 0:
            logger.warning("No movies found within epsilon ball")
            return pd.DataFrame(
                columns=["movie_id", "title", "year", "distance", "similarity", "rank"]
            )

        results = []
        for rank, (idx, dist, sim) in enumerate(
            zip(indices, distances, similarities), 1
        ):
            movie_id = filtered_movie_ids[idx]
            movie_info = movie_data[movie_data["movie_id"] == movie_id]

            if not movie_info.empty:
                title = movie_info.iloc[0]["title"]
                year = movie_info.iloc[0].get("year", None)
            else:
                title = "Unknown"
                year = None

            results.append(
                {
                    "movie_id": movie_id,
                    "title": title,
                    "year": year,
                    "distance": dist,
                    "similarity": sim,
                    "rank": rank,
                }
            )

        results_df = pd.DataFrame(results)

        logger.info(f"\n{'=' * 60}")
        logger.info("Results summary:")
        logger.info(f"Total movies in epsilon ball: {len(results_df)}")
        logger.info(
            f"Distance range: {results_df['distance'].min():.6f} - {results_df['distance'].max():.6f}"
        )
        if "year" in results_df.columns and results_df["year"].notna().any():
            logger.info(
                f"Year range: {int(results_df['year'].min())} - {int(results_df['year'].max())}"
            )
        logger.info(f"{'=' * 60}")

        return results_df

    except Exception as e:
        logger.error(f"Error in epsilon ball analysis: {e}")
        raise


def main(
    anchor_qids: list = None,
    epsilon: float = 0.3,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    data_dir: str = DATA_DIR,
    csv_path: str = CSV_PATH,
    anchor_method: str = "average",
    exclude_anchors: bool = True,
    plot_over_time: bool = True,
    plot_distance_dist: bool = True,
    compare_with_random: bool = False,
    output_dir: str = None,
):
    """
    Main function to run epsilon ball analysis.

    Parameters:
    - anchor_qids: List of movie QIDs to use as anchors (e.g., ["Q4941"] for James Bond)
    - epsilon: Maximum cosine distance threshold (default: 0.3)
    - start_year: First year to filter movies (default: 1930)
    - end_year: Last year to filter movies (default: 2024)
    - data_dir: Directory containing the final embedding files
    - csv_path: Path to final_dataset.csv
    - anchor_method: Method to combine anchor embeddings ("average" or "medoid")
    - exclude_anchors: Whether to exclude anchor movies from results
    - plot_over_time: Whether to create plot of movies over time
    - plot_distance_dist: Whether to create distance distribution plot
    - compare_with_random: Whether to compare with control group (mean of entire ensemble) (default: False)
    - output_dir: Directory to save plots (if None, uses current directory)
    """
    if anchor_qids is None:
        anchor_qids = ["Q4941"]
        logger.info("Using default anchor: James Bond (Q4941)")

    logger.info("Loading embeddings...")
    all_embeddings, all_movie_ids = load_final_dense_embeddings(data_dir, verbose=False)

    if len(all_movie_ids) == 0:
        raise ValueError(f"No embeddings found in {data_dir}")

    logger.info(f"Total movies with embeddings: {len(all_movie_ids)}")
    logger.info(f"Embedding shape: {all_embeddings.shape}")

    logger.info("Loading movie metadata...")
    movie_data = load_final_dataset(csv_path, verbose=False)

    if movie_data.empty:
        raise ValueError(f"No movie data found in {csv_path}")

    if "year" in movie_data.columns:
        movie_data = movie_data[
            (movie_data["year"] >= start_year) & (movie_data["year"] <= end_year)
        ].copy()
        logger.info(
            f"Filtered to {len(movie_data)} movies between {start_year} and {end_year}"
        )

    valid_movie_ids = set(movie_data["movie_id"].values)
    valid_indices = np.array(
        [i for i, mid in enumerate(all_movie_ids) if mid in valid_movie_ids]
    )

    if len(valid_indices) == 0:
        raise ValueError(
            f"No movies found in the specified year range ({start_year}-{end_year})"
        )

    filtered_embeddings = all_embeddings[valid_indices]
    filtered_movie_ids = all_movie_ids[valid_indices]

    logger.info(
        f"Filtered to {len(filtered_movie_ids)} movies with embeddings in year range"
    )

    results_df = analyze_epsilon_ball(
        anchor_qids=anchor_qids,
        epsilon=epsilon,
        filtered_embeddings=filtered_embeddings,
        filtered_movie_ids=filtered_movie_ids,
        movie_data=movie_data,
        anchor_method=anchor_method,
        exclude_anchors=exclude_anchors,
    )

    random_results_df = None
    random_distances_list = []
    if compare_with_random:
        logger.info("Computing mean embedding of entire ensemble as control group...")

        embeddings_hash = compute_embeddings_hash(
            filtered_embeddings=filtered_embeddings,
            start_year=start_year,
            end_year=end_year,
        )

        mean_embedding, cache_found = load_cached_mean_embedding(
            cache_dir=CACHE_DIR,
            embeddings_hash=embeddings_hash,
        )

        if not cache_found:
            logger.info("Computing mean embedding (this may take a while)...")
            mean_embedding = np.mean(filtered_embeddings, axis=0, keepdims=True)
            logger.info(
                f"Mean embedding computed from {len(filtered_embeddings)} movies"
            )

            save_cached_mean_embedding(
                mean_embedding=mean_embedding,
                cache_dir=CACHE_DIR,
                embeddings_hash=embeddings_hash,
                n_movies=len(filtered_embeddings),
            )
        else:
            logger.info(
                f"Using cached mean embedding (computed from {len(filtered_embeddings)} movies)"
            )

        logger.info(
            f"Finding movies within epsilon ball (epsilon={epsilon}) around mean embedding..."
        )
        indices, distances, similarities = find_movies_in_epsilon_ball(
            embeddings_corpus=filtered_embeddings,
            anchor_embedding=mean_embedding,
            movie_ids=filtered_movie_ids,
            epsilon=epsilon,
            exclude_anchor_ids=None,
        )

        logger.info(
            f"Found {len(indices)} movies within epsilon ball of mean embedding"
        )

        results = []
        for rank, (idx, dist, sim) in enumerate(
            zip(indices, distances, similarities), 1
        ):
            movie_id = filtered_movie_ids[idx]
            movie_info = movie_data[movie_data["movie_id"] == movie_id]

            if not movie_info.empty:
                title = movie_info.iloc[0]["title"]
                year = movie_info.iloc[0].get("year", None)
            else:
                title = "Unknown"
                year = None

            results.append(
                {
                    "movie_id": movie_id,
                    "title": title,
                    "year": year,
                    "distance": dist,
                    "similarity": sim,
                    "rank": rank,
                }
            )

        random_results_df = pd.DataFrame(results)

        if not random_results_df.empty and "distance" in random_results_df.columns:
            random_distances_list.append(random_results_df["distance"].values)

    if (
        compare_with_random
        and random_results_df is not None
        and not random_results_df.empty
        and not results_df.empty
    ):
        logger.info(f"\n{'=' * 60}")
        logger.info("Kolmogorov-Smirnov Test Results")
        logger.info(f"{'=' * 60}")

        if random_distances_list and "distance" in results_df.columns:
            all_random_distances = np.concatenate(random_distances_list)
            # Use ALL movies in epsilon ball for KS test (not limited)
            anchor_distances = results_df["distance"].values

            logger.info(
                f"Using {len(anchor_distances)} movies from epsilon ball for distance K-S test"
            )
            logger.info(
                f"Using {len(all_random_distances)} movies from control group for distance K-S test"
            )

            try:
                ks_stat_dist, p_value_dist = kolmogorov_smirnov_test(
                    anchor_distances, all_random_distances
                )
                interpretation_dist = interpret_ks_test(
                    ks_stat_dist,
                    p_value_dist,
                    sample_size_1=len(anchor_distances),
                    sample_size_2=len(all_random_distances),
                )

                logger.info("\n1. Distance Distribution Comparison:")
                logger.info(f"   K-S Statistic: {ks_stat_dist:.6f}")

                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    anchor_names_str = get_anchor_names_string(anchor_qids, movie_data)
                    anchor_names_str = truncate_filename_component(anchor_names_str)
                    ks_plot_path = os.path.join(
                        output_dir,
                        f"ks_test_distances_{anchor_names_str}_eps{epsilon:.2f}.png",
                    )
                    plot_ks_test_cdf(
                        anchor_distances,
                        all_random_distances,
                        ks_stat_dist,
                        p_value_dist,
                        output_path=ks_plot_path,
                        title=f"K-S Test: Distance Distributions (ε={epsilon})",
                        interpretation=interpretation_dist,
                    )
            except Exception as e:
                logger.warning(f"Could not perform distance K-S test: {e}")

        if random_results_df is not None and not random_results_df.empty:
            # Use ALL movies in epsilon ball for temporal analysis (not limited)
            anchor_df_with_year = results_df[results_df["year"].notna()].copy()
            anchor_df_with_year["year"] = anchor_df_with_year["year"].astype(int)
            anchor_year_counts = anchor_df_with_year["year"].value_counts().sort_index()

            random_df_with_year = random_results_df[
                random_results_df["year"].notna()
            ].copy()
            random_df_with_year["year"] = random_df_with_year["year"].astype(int)
            random_year_counts = random_df_with_year["year"].value_counts().sort_index()

            logger.info(
                f"Using {len(anchor_df_with_year)} movies from epsilon ball for temporal K-S test"
            )
            logger.info(
                f"Using {len(random_df_with_year)} movies from control group for temporal K-S test"
            )

            all_years = sorted(
                set(anchor_year_counts.index) | set(random_year_counts.index)
            )
            anchor_aligned = pd.Series(0, index=all_years)
            random_aligned = pd.Series(0, index=all_years)
            anchor_aligned.loc[anchor_year_counts.index] = anchor_year_counts.values
            random_aligned.loc[random_year_counts.index] = random_year_counts.values

            try:
                ks_stat_temp, p_value_temp = kolmogorov_smirnov_test_temporal(
                    anchor_aligned.values,
                    random_aligned.values,
                    all_years,
                )
                anchor_temporal_size = int(anchor_aligned.sum())
                random_temporal_size = int(random_aligned.sum())

                interpretation_temp = interpret_ks_test(
                    ks_stat_temp,
                    p_value_temp,
                    sample_size_1=anchor_temporal_size,
                    sample_size_2=random_temporal_size,
                )

                logger.info("\n2. Temporal Distribution Comparison:")
                logger.info(f"   K-S Statistic: {ks_stat_temp:.6f}")

                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    anchor_names_str = get_anchor_names_string(anchor_qids, movie_data)
                    anchor_names_str = truncate_filename_component(anchor_names_str)
                    ks_plot_path = os.path.join(
                        output_dir,
                        f"ks_test_temporal_{anchor_names_str}_eps{epsilon:.2f}.png",
                    )
                    plot_ks_test_temporal_cdf(
                        anchor_year_counts,
                        random_year_counts,
                        ks_stat_temp,
                        p_value_temp,
                        output_path=ks_plot_path,
                        title=f"K-S Test: Temporal Distributions (ε={epsilon})",
                        interpretation=interpretation_temp,
                    )
            except Exception as e:
                logger.warning(f"Could not perform temporal K-S test: {e}")

        logger.info(f"\n{'=' * 60}")

    logger.info(f"\n{'=' * 60}")
    logger.info("Top 20 movies in epsilon ball:")
    logger.info(f"{'=' * 60}")
    logger.info("")
    for _, row in results_df.head(20).iterrows():
        logger.info(
            f"Rank {row['rank']}: {row['title']} ({row['movie_id']}) - "
            f"Distance: {row['distance']:.6f}, Year: {row.get('year', 'N/A')}"
        )

    if plot_over_time and not results_df.empty:
        logger.info(
            f"Plotting movies over time using ALL {len(results_df)} movies from epsilon ball"
        )
        output_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            anchor_names_str = get_anchor_names_string(anchor_qids, movie_data)
            anchor_names_str = truncate_filename_component(anchor_names_str)
            output_path = os.path.join(
                output_dir,
                f"epsilon_ball_over_time_{anchor_names_str}_eps{epsilon:.2f}.png",
            )
        # Use ALL movies in epsilon ball (not limited)
        plot_movies_over_time(
            results_df,
            output_path=output_path,
            title=f"Movies in Epsilon Ball (ε={epsilon}) Over Time",
            random_results_df=random_results_df,
        )

    if plot_distance_dist and not results_df.empty:
        logger.info(
            f"Plotting distance distribution using ALL {len(results_df)} movies from epsilon ball"
        )
        output_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            anchor_names_str = get_anchor_names_string(anchor_qids, movie_data)
            anchor_names_str = truncate_filename_component(anchor_names_str)
            output_path = os.path.join(
                output_dir,
                f"epsilon_ball_distance_dist_{anchor_names_str}_eps{epsilon:.2f}.png",
            )
        # Use ALL movies in epsilon ball (not limited)
        plot_distance_distribution(
            results_df,
            output_path=output_path,
            title=f"Distance Distribution in Epsilon Ball (ε={epsilon})",
        )

    return results_df


if __name__ == "__main__":
    # ["Q18602670", "Q212145", "Q207916", "Q151904", "Q591272", "Q19089", "Q181540", "Q107914", "Q320423", "Q332368", "Q21534241", "Q30931", "Q106440"] # James Bond movies

    # ['Q1187607', 'Q488655', 'Q174992', 'Q785406', 'Q22350712'] # Edge of Tomorrow, Groundhog Day, 12:01, The Time Traveler's Wife, Before I Fall

    # ['Q507994', 'Q1140085', 'Q62730', 'Q244929', 'Q662342', 'Q1320705', 'Q1263897', 'Q1124501', 'Q1469774', 'Q747395', 'Q27888483'] # The Hunt for Red October, Crimson Tide, Das Boot, The Bedford Incident, K-19: The Widowmaker, Ice Station Zebra, The Enemy Below, By Dawn's Early Light, On the Beach, Hunter Killer

    # ['Q1366386', 'Q25136484', 'Q2171744', 'Q2364210', 'Q1490812', 'Q2746506', 'Q7617650', 'Q50650165'] # It, It Chapter Two, Killer Klowns from Outer Space, Clownhouse, Gacy, House of 1000 Corpses, Drive-Thru, Stitches, Terrifier

    # ['Q220735', 'Q261209', 'Q657079', 'Q7763422', 'Q110206', 'Q633171', 'Q4186834', 'Q1709419', 'Q760926', 'Q613485'] # The French Connection, Bullitt, Serpico, The Seven-Ups, Dirty Harry, The Taking of Pelham One Two Three, Prince of the City, Fort Apache, The Bronx, Cruising, To Live and Die in L.A.

    # ['Q192724', 'Q3820040', 'Q192724', 'Q466611', 'Q205028', 'Q217020', 'Q275120', 'Q3985737', 'Q494985', 'Q182218', 'Q209538', 'Q1201853', 'Q1765358', 'Q14171368', 'Q18407657', 'Q18406872', 'Q5887360', 'Q20001199', 'Q23010088', 'Q22665878', 'Q23780734', 'Q23780914', 'Q23781155', 'Q27985819'] # Iron Man, Iron Man 2, Iron Man 3, The Incredible Hulk, Thor, Captain America: The First Avenger, The Avengers, Iron Man 3, Thor: The Dark World, Captain America: The Winter Soldier, Avengers: Age of Ultron, Captain America: Civil War, Doctor Strange, Guardians of the Galaxy, Guardians of the Galaxy Vol. 2, Spider-Man: Homecoming, Thor: Ragnarok, Black Panther, Avengers: Infinity War, Avengers: Endgame, Spider-Man: Far From Home

    # ['Q221384', 'Q183066', 'Q152531', 'Q16970789', 'Q3258993', 'Q19865453', 'Q632328', 'Q848785', 'Q578312', 'Q1394447', 'Q17093105', 'Q20751325', 'Q63927168', 'Q21463782'] # Black Hawk Down, The Hurt Locker, Zero Dark Thirty, American Sniper, Lone Survivor, 13 Hours: The Secret Soldiers of Benghazi, Green Zone, Jarhead, Body of Lies, Restrepo, Korengal, Hyena Road, Kajaki (also released as Kilo Two Bravo), The Outpost, War Machine

    # most average movies # ['Q26683632', 'Q5932706', 'Q12671094', 'Q444057', 'Q17071466', 'Q554539', 'Q2549142', 'Q6872502', 'Q105441001', 'Q23755528']

    # ['Q41483', 'Q168154', 'Q245208', 'Q20092609', 'Q276769', 'Q1008351', 'Q50714', 'Q3208286', 'Q7596837', 'Q104137', 'Q8061777', 'Q47352417', 'Q994481', 'Q603263', 'Q326114', 'Q232000', 'Q153677', 'Q76479', 'Q241811', 'Q19069', 'Q19983487', 'Q247130', 'Q746029'] # The Good, the Bad and the Ugly, Once Upon a Time in the West, High Noon, The Searchers, Rio Bravo, Stagecoach, Unforgiven, The Wild Bunch, True Grit, Butch Cassidy and the Sundance Kid, For a Few Dollars More, A Fistful of Dollars, Django, The Magnificent Seven, 3:10 to Yuma

    # ["Q103474", "Q184843", "Q21500755", "Q162255", "Q170564", "Q16635326", "Q788822", "Q131191955", "Q221113", "Q83495", "Q189600", "Q207536", "Q200572", "Q504697", "Q626483", "Q18954", "Q244604", "Q1066948", "Q22575835", "Q10384115", "Q3549863", "Q30611788", "Q26751"] # AI Movies

    # ["Q103569", "Q104814", "Q20430699", "Q200804", "Q720357", "Q210756", "Q909749", "Q11621", "Q320588", "Q20382729", "Q1657967", "Q105387", "Q202028", "Q201819", "Q425992", "Q187154", "Q45386", "Q598818", "Q22432", "Q3205861", "Q336517", "Q25136228", "Q5164779", "Q270215", "Q15803822"] # Alien Movies

    results = main(
        anchor_qids=[
            "Q18602670",
            "Q212145",
            "Q207916",
            "Q151904",
            "Q591272",
            "Q19089",
            "Q181540",
            "Q107914",
            "Q320423",
            "Q332368",
            "Q21534241",
            "Q30931",
            "Q106440",
        ],
        epsilon=0.32,
        start_year=1930,
        end_year=2024,
        anchor_method="average",  # or "medoid"
        exclude_anchors=True,
        compare_with_random=True,
        plot_over_time=True,
        plot_distance_dist=True,
        output_dir=f"{BASE_DIR}/figures/epsilon_ball_analysis",
    )

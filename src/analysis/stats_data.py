"""
Statistics Analysis for Movie Data

This script analyzes all movie data files and provides:
- Overall statistics (total movies, field completeness percentages)
- Per-year statistics
- Plot-specific statistics (count, average length, histogram)
"""

import os
import sys
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import seaborn as sns
from scipy.stats import cosine

from itertools import combinations
from scipy.spatial.distance import cosine as another_cosine

from analysis.chunking.calculations import compute_cosine_distance, calculate_drift_vector
from scipy.spatial.distance import cosine as cos_dist


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
try:
    # This file is in src/analysis/, so go up two levels to get project root
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)
except NameError:
    BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, 'data')


def find_year_files(data_dir: str) -> Dict[int, str]:
    """
    Find all CSV files matching the pattern wikidata_movies_YYYY.csv
    
    Returns:
        Dictionary mapping year to file path
    """
    year_files = {}
    data_path = Path(data_dir)
    
    for csv_file in data_path.glob('wikidata_movies_*.csv'):
        # Extract year from filename (e.g., wikidata_movies_1950.csv -> 1950)
        try:
            year_str = csv_file.stem.split('_')[-1]
            # Handle files like "1950_to_2024" by taking first year
            if 'to' in year_str:
                year_str = year_str.split('_to_')[0]
            year = int(year_str)
            if year not in year_files:  # Prefer single year files over range files
                year_files[year] = str(csv_file)
        except (ValueError, IndexError):
            continue
    
    return year_files


def load_all_data(data_dir: str) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """
    Load all year-specific CSV files.
    
    Returns:
        Tuple of (combined DataFrame, dictionary of year -> DataFrame)
    """
    year_files = find_year_files(data_dir)
    
    if not year_files:
        logger.warning(f"No year-specific CSV files found in {data_dir}")
        return pd.DataFrame(), {}
    
    logger.info(f"Found {len(year_files)} year files")
    
    year_dataframes = {}
    all_dataframes = []
    
    for year in sorted(year_files.keys()):
        file_path = year_files[year]
        try:
            df = pd.read_csv(file_path, dtype=str, low_memory=False)
            df['year'] = year  # Ensure year column is set
            year_dataframes[year] = df
            all_dataframes.append(df)
            logger.info(f"Year {year}: Loaded {len(df)} movies from {Path(file_path).name}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            continue
    
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        logger.info(f"Combined dataset: {len(combined_df)} total movies")
    else:
        combined_df = pd.DataFrame()
    
    return combined_df, year_dataframes


def calculate_field_statistics(df: pd.DataFrame, field: str) -> Dict[str, float]:
    """
    Calculate statistics for a specific field.
    
    Returns:
        Dictionary with count, percentage, and average length (if applicable)
    """
    if field not in df.columns:
        return {
            'total': len(df),
            'present': 0,
            'percentage': 0.0,
            'missing': len(df)
        }
    
    # Count non-null and non-empty values
    if df[field].dtype == 'object':
        # For string fields, check for non-null and non-empty strings
        present = df[field].notna() & (df[field].astype(str).str.strip() != '') & (df[field].astype(str) != 'nan')
    else:
        present = df[field].notna()
    
    count_present = present.sum()
    count_missing = len(df) - count_present
    percentage = (count_present / len(df) * 100) if len(df) > 0 else 0.0
    
    stats = {
        'total': len(df),
        'present': int(count_present),
        'percentage': round(percentage, 2),
        'missing': int(count_missing)
    }
    
    # For string fields, calculate average length
    if df[field].dtype == 'object' and count_present > 0:
        lengths = df[present][field].astype(str).str.len()
        stats['avg_length'] = round(lengths.mean(), 2)
        stats['min_length'] = int(lengths.min()) if len(lengths) > 0 else 0
        stats['max_length'] = int(lengths.max()) if len(lengths) > 0 else 0
    
    return stats


def print_field_statistics(df: pd.DataFrame, title: str = "Statistics"):
    """
    Print statistics for all fields in the DataFrame.
    """
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)
    print(f"Total movies: {len(df)}")
    print("\nField Completeness:")
    print("-" * 80)
    print(f"{'Field':<25} {'Present':<10} {'Missing':<10} {'Percentage':<12} {'Avg Length':<12}")
    print("-" * 80)
    
    # Get all columns except 'year' (we'll show it separately if needed)
    fields = [col for col in df.columns if col != 'year']
    
    for field in fields:
        stats = calculate_field_statistics(df, field)
        avg_len = stats.get('avg_length', 'N/A')
        print(f"{field:<25} {stats['present']:<10} {stats['missing']:<10} {stats['percentage']:<12.2f}% {str(avg_len):<12}")
    
    print("-" * 80)


def print_per_year_statistics(year_dataframes: Dict[int, pd.DataFrame]):
    """
    Print statistics per year for key fields.
    """
    print("\n" + "=" * 80)
    print("PER-YEAR STATISTICS")
    print("=" * 80)
    
    # Key fields to show
    key_fields = ['title', 'director', 'actors', 'genre', 'plot', 'popularity', 
                  'vote_average', 'vote_count', 'imdb_id', 'tmdb_id', 'duration']
    
    # Print header
    header = f"{'Year':<8} {'Total':<8}"
    for field in key_fields:
        header += f" {field[:8]:<8}"
    print(header)
    print("-" * len(header))
    
    for year in sorted(year_dataframes.keys()):
        df = year_dataframes[year]
        if df.empty:
            continue
        
        row = f"{year:<8} {len(df):<8}"
        for field in key_fields:
            stats = calculate_field_statistics(df, field)
            row += f" {stats['percentage']:>6.1f}%"
        print(row)
    
    print("-" * len(header))


def analyze_plots(df: pd.DataFrame, output_dir: str = None):
    """
    Analyze plot/summary data: count, average length, and create histogram.
    Checks for both 'plot' and 'summary' columns.
    """
    print("\n" + "=" * 80)
    print("PLOT/SUMMARY ANALYSIS")
    print("=" * 80)
    
    total_movies = len(df)
    print(f"Total movies: {total_movies}")
    
    # Check for plot column first (preferred)
    has_plot_column = 'plot' in df.columns
    has_summary_column = 'summary' in df.columns
    
    if not has_plot_column and not has_summary_column:
        print("No 'plot' or 'summary' column found in data")
        return
    
    # Use plot if available, otherwise use summary
    if has_plot_column:
        text_column = 'plot'
        text_type = 'Plot'
        text_type_plural = 'plots'
    else:
        text_column = 'summary'
        text_type = 'Summary'
        text_type_plural = 'summaries'
    
    # Check for plots/summaries (non-null and non-empty)
    has_text = df[text_column].notna() & (df[text_column].astype(str).str.strip() != '') & (df[text_column].astype(str) != 'nan')
    text_count = has_text.sum()
    
    print(f"Movies with {text_type_plural}: {text_count} ({text_count/total_movies*100:.2f}%)")
    print(f"Movies without {text_type_plural}: {total_movies - text_count} ({(total_movies - text_count)/total_movies*100:.2f}%)")
    
    # Also show summary stats if both columns exist
    if has_plot_column and has_summary_column:
        has_summary = df['summary'].notna() & (df['summary'].astype(str).str.strip() != '') & (df['summary'].astype(str) != 'nan')
        summary_count = has_summary.sum()
        print(f"\nMovies with summaries: {summary_count} ({summary_count/total_movies*100:.2f}%)")
    
    if text_count == 0:
        print(f"No {text_type_plural} available for analysis")
        return
    
    # Calculate text lengths
    texts = df[has_text][text_column].astype(str)
    text_lengths = texts.str.len()
    
    print(f"\n{text_type} Length Statistics:")
    print(f"  Average length: {text_lengths.mean():.2f} characters")
    print(f"  Median length: {text_lengths.median():.2f} characters")
    print(f"  Min length: {text_lengths.min()} characters")
    print(f"  Max length: {text_lengths.max()} characters")
    print(f"  Standard deviation: {text_lengths.std():.2f} characters")
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    plt.hist(text_lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel(f'{text_type} Length (characters)', fontsize=12)
    plt.ylabel('Number of Movies', fontsize=12)
    plt.title(f'Distribution of {text_type} Lengths (n={text_count} movies with {text_type_plural})', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add vertical line for mean
    mean_length = text_lengths.mean()
    plt.axvline(mean_length, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_length:.0f} chars')
    plt.legend()
    
    # Save histogram
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        histogram_path = os.path.join(output_dir, f'{text_column.lower()}_lengths_histogram.png')
        plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
        print(f"\nHistogram saved to: {histogram_path}")
    else:
        histogram_path = os.path.join(DATA_DIR, f'{text_column.lower()}_lengths_histogram.png')
        plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
        print(f"\nHistogram saved to: {histogram_path}")
    
    plt.close()


def save_detailed_statistics(df: pd.DataFrame, year_dataframes: Dict[int, pd.DataFrame], 
                            output_file: str = None):
    """
    Save detailed statistics to a CSV file.
    """
    if output_file is None:
        output_file = os.path.join(DATA_DIR, 'data_statistics.csv')
    
    # Collect statistics for all fields
    fields = [col for col in df.columns if col != 'year']
    stats_rows = []
    
    # Overall statistics
    for field in fields:
        stats = calculate_field_statistics(df, field)
        stats_rows.append({
            'scope': 'Overall',
            'year': 'All',
            'field': field,
            'total_movies': stats['total'],
            'present': stats['present'],
            'missing': stats['missing'],
            'percentage': stats['percentage'],
            'avg_length': stats.get('avg_length', None),
            'min_length': stats.get('min_length', None),
            'max_length': stats.get('max_length', None)
        })
    
    # Per-year statistics
    for year in sorted(year_dataframes.keys()):
        df_year = year_dataframes[year]
        if df_year.empty:
            continue
        
        for field in fields:
            stats = calculate_field_statistics(df_year, field)
            stats_rows.append({
                'scope': 'Per-Year',
                'year': year,
                'field': field,
                'total_movies': stats['total'],
                'present': stats['present'],
                'missing': stats['missing'],
                'percentage': stats['percentage'],
                'avg_length': stats.get('avg_length', None),
                'min_length': stats.get('min_length', None),
                'max_length': stats.get('max_length', None)
            })
    
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(output_file, index=False)
    print(f"\nDetailed statistics saved to: {output_file}")


def main(data_dir: str = None, save_stats: bool = True, save_histogram: bool = True):
    """
    Main function to run all statistics analysis.
    
    Args:
        data_dir: Directory containing CSV files (defaults to DATA_DIR)
        save_stats: Whether to save detailed statistics to CSV
        save_histogram: Whether to save plot length histogram
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    logger.info(f"Loading data from: {data_dir}")
    
    # Load all data
    combined_df, year_dataframes = load_all_data(data_dir)
    
    if combined_df.empty:
        logger.error("No data found to analyze")
        return
    
    # Overall statistics
    print_field_statistics(combined_df, "OVERALL STATISTICS")
    
    # Per-year statistics
    print_per_year_statistics(year_dataframes)
    
    # Plot analysis
    analyze_plots(combined_df, output_dir=data_dir if save_histogram else None)
    
    # Save detailed statistics
    if save_stats:
        save_detailed_statistics(combined_df, year_dataframes)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


def calculate_genre_drift(df_filtered):
    """
    Calculates average embeddings per year and genre, and then computes
    the annual drift distance (cosine distance) and cumulative change.
    """
    # Calculate average embedding per year and genre
    grouped_embeddings = df_filtered.groupby(['year', 'new_genre'])['embedding'].apply(
        lambda x: np.mean(np.vstack(x), axis=0)
    )
    group_df = grouped_embeddings.reset_index(name='avg_embedding')
    group_df = group_df.sort_values(by=['new_genre', 'year']).copy()

    # Shift embeddings to align t and t+1 for drift calculation
    group_df['next_avg_embedding'] = group_df.groupby('new_genre')['avg_embedding'].shift(-1)

    # Calculate drift distance (cosine distance between consecutive average embeddings)
    group_df['drift_distance'] = group_df.apply(compute_cosine_distance, axis=1)
    drift_df = group_df.dropna(subset=['drift_distance']).copy()

    # Calculate cumulative change per genre
    drift_df['cumulative_change'] = drift_df.groupby('new_genre')['drift_distance'].cumsum()

    return drift_df, group_df

def prepare_heatmap_data(group_df, target_genre, bin_size=5):
    """
    Groups average embeddings into year intervals, calculates the drift vector 
    between intervals, and formats the result for a heatmap.
    """
    genre_df = group_df[group_df['new_genre'] == target_genre].copy()

    # Group by year intervals
    genre_df['year_interval'] = (genre_df['year'] // bin_size) * bin_size
    interval_df = genre_df.groupby('year_interval')['avg_embedding'].apply(
        lambda x: np.mean(np.vstack(x), axis=0)
    ).reset_index(name='avg_embedding')

    # Calculate drift vectors
    interval_df['next_avg_embedding'] = interval_df['avg_embedding'].shift(-1)
    interval_df['drift_vector'] = interval_df.apply(calculate_drift_vector, axis=1)
    final_drift_df = interval_df.dropna(subset=['drift_vector']).copy()

    drift_matrix = np.vstack(final_drift_df['drift_vector'].values)
    interval_labels = final_drift_df['year_interval'].map(lambda y: f'{y} -> {y + bin_size}')

    heatmap_df = pd.DataFrame(
        drift_matrix,
        index=interval_labels,
        columns=[f'Dim_{i}' for i in range(drift_matrix.shape[1])]
    )
    return heatmap_df, bin_size


def calculate_binned_inter_genre_distance(group_df, bin_size):
    """
    Calculates the cosine distance between the average embeddings of all
    unique genre pairs for every time interval (bin_size), indicating convergence/divergence.
    """
    df_binned = group_df.copy()
    df_binned['year_interval'] = (df_binned['year'] // bin_size) * bin_size

    # Calculate average embedding per interval and genre
    position_vectors_df = df_binned.groupby(['year_interval', 'single_genre'])['avg_embedding'].apply(
        lambda x: np.mean(np.vstack(x), axis=0)
    ).reset_index(name='avg_embedding')

    # Identify all unique genre pairs
    unique_genres = position_vectors_df['single_genre'].unique()
    genre_pairs = list(combinations(unique_genres, 2))
    distance_results = []

    # Pivot to easily compare vectors by interval
    pivot_table = position_vectors_df.pivot_table(
        index='year_interval',
        columns='single_genre',
        values='avg_embedding',
        aggfunc='first'
    )

    for g1, g2 in genre_pairs:
        vectors_g1 = pivot_table[g1]
        vectors_g2 = pivot_table[g2]

        # Filter intervals where both genres have data
        comparison_df = pd.DataFrame({
            'v1': vectors_g1,
            'v2': vectors_g2
        }).dropna()

        if comparison_df.empty:
            continue

        # Apply cosine distance function to each row (interval)
        distances = comparison_df.apply(
            # Using scipy's cosine function as in your original code
            lambda row: cosine(row['v1'], row['v2']), axis=1
        )

        # Format results
        for interval, distance in distances.items():
            distance_results.append({
                'Genre_A': g1,
                'Genre_B': g2,
                'Year_Interval_Start': interval,
                'Cosine_Distance': distance
            })

    if not distance_results:
        return pd.DataFrame()

    convergence_df = pd.DataFrame(distance_results)
    convergence_df = convergence_df.sort_values(
        by=['Genre_A', 'Genre_B', 'Year_Interval_Start']
    ).reset_index(drop=True)

    return convergence_df


def calculate_genre_convergence(group_df, bin_size=5, target_genres=None):
    """
    Calculate cosine distance between genres to analyse convergence / divergence
    """

    df_filtered = group_df.copy()
    df_filtered['year_interval'] = (df_filtered['year'] // bin_size) * bin_size
    position_vectors_df = df_filtered.groupby(['year_interval', 'new_genre'])['avg_embedding'].apply(
        lambda x: np.mean(np.vstack(x), axis=0)
    ).reset_index(name='avg_embedding')

    # Identify genre pairs
    unique_genres = position_vectors_df['new_genre'].unique()
    genre_pairs = list(combinations(unique_genres, 2))
    distance_results = []

    # Easy lookup: {interval: {genre: vector, ...}, ...}
    pivot_table = position_vectors_df.pivot_table(
        index='year_interval',
        columns='new_genre',
        values='avg_embedding',
        aggfunc='first'
    )

    for g1, g2 in genre_pairs:
        vectors_g1 = pivot_table[g1]
        vectors_g2 = pivot_table[g2]

        # Filter only intervals in which both genres have data
        comparison_df = pd.DataFrame({
            'v1': vectors_g1,
            'v2': vectors_g2
        }).dropna()

        if comparison_df.empty:
            continue

        distances = comparison_df.apply(
            lambda row: another_cosine(row['v1'], row['v2']), axis=1
        )
        for interval, distance in distances.items():
            distance_results.append({
                'Genre_A': g1,
                'Genre_B': g2,
                'Year_Interval_Start': interval,
                'Cosine_Distance': distance
            })

    if not distance_results:
        return pd.DataFrame()

    convergence_df = pd.DataFrame(distance_results)

    # Order by pair and time
    convergence_df = convergence_df.sort_values(by=['Genre_A', 'Genre_B', 'Year_Interval_Start']).reset_index(drop=True)

    return convergence_df


def unpivot_convergence_df(convergence_df):
    """
    A,B comparison -> A vs B && B vs A
    """
    # Version A main genre
    df_a = convergence_df.rename(columns={'Genre_A': 'Genre', 'Genre_B': 'Compared_To'})
    df_a = df_a[['Year_Interval_Start', 'Genre', 'Compared_To', 'Cosine_Distance']]

    # Version B main genre
    df_b = convergence_df.rename(columns={'Genre_B': 'Genre', 'Genre_A': 'Compared_To'})
    df_b = df_b[['Year_Interval_Start', 'Genre', 'Compared_To', 'Cosine_Distance']]

    return pd.concat([df_a, df_b])

def unpivot_distance_for_plotting(convergence_df):
    """
    Unpivots the convergence DataFrame so each comparison (A vs B)
    appears twice (once for A and once for B), allowing for mean calculation per genre.
    """
    df_a = convergence_df.rename(columns={'Genre_A': 'Genre', 'Genre_B': 'Compared_To'})
    df_a = df_a[['Year_Interval_Start', 'Genre', 'Compared_To', 'Cosine_Distance']]

    df_b = convergence_df.rename(columns={'Genre_B': 'Genre', 'Genre_A': 'Compared_To'})
    df_b = df_b[['Year_Interval_Start', 'Genre', 'Compared_To', 'Cosine_Distance']]

    return pd.concat([df_a, df_b])



def plot_genre_drift(df, y_column, title, y_label):
    """Generates a time-series line plot for an individual genre drift metric."""
    sns.set_theme(style="whitegrid")
    df['year'] = pd.to_numeric(df['year'])

    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df,
        x='year',
        y=y_column,
        hue='new_genre',
        marker='o',
        linewidth=1.5
    )
    plt.title(title, fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()
    
def plot_standard_heatmap(heatmap_df, target_genre, bin_size):
    """Generates and displays a standard heatmap of the drift vectors."""
    if heatmap_df is None: return

    plt.figure(figsize=(16, 8))
    sns.heatmap(
        heatmap_df.T, cmap='coolwarm', center=0,
        cbar_kws={'label': 'Drift value'}, yticklabels=False
    )
    plt.title(f'Drift of genre "{target_genre}" ({bin_size} Years) - Standard', fontsize=16)
    plt.xlabel('Drift intervals', fontsize=12)
    plt.ylabel('Embedding dimensions', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_clustermap(heatmap_df, target_genre, bin_size):
    """Generates and displays a clustered heatmap of the drift vectors."""
    if heatmap_df is None: return

    sns.clustermap(
        heatmap_df.T, cmap='coolwarm', center=0,
        row_cluster=True, col_cluster=False, figsize=(16, 12),
        cbar_kws={'label': 'Drift value'}, yticklabels=False
    )
    plt.suptitle(f'Drift of genre "{target_genre}" ({bin_size} Years) - Dimensional Clustering', fontsize=16)
    plt.show()


def plot_mean_convergence(convergence_df, bin_size):
    """
    Plots the mean distance of each genre relative to the average of all others.
    """
    if convergence_df.empty:
        print("No mean convergence data to plot.")
        return

    unpivoted_df = unpivot_distance_for_plotting(convergence_df)
    mean_distance_df = unpivoted_df.groupby(['Year_Interval_Start', 'Genre'])['Cosine_Distance'].mean().reset_index(name='Mean_Distance')

    plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=mean_distance_df,
        x='Year_Interval_Start',
        y='Mean_Distance',
        hue='Genre',
        marker='o',
        linewidth=2
    )

    plt.title(f'Mean Cosine Distance of Each Genre vs. the Rest of the Group ({bin_size} Years)', fontsize=16)
    plt.xlabel('Interval Start Year', fontsize=12)
    plt.ylabel('Mean Cosine Distance', fontsize=12)
    plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_pairwise_convergence(convergence_df, target_genres=None):
    """
    Subplot per each genre
    """

    # Desplegar el DF para facilitar el filtrado por género principal
    unpivoted_df = unpivot_convergence_df(convergence_df)

    if target_genres is None:
        target_genres = unpivoted_df['Genre'].unique()

    # Calcular el número de subplots necesarios
    n_genres = len(target_genres)
    cols = 3
    rows = (n_genres + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True, sharey=True)
    axes = axes.flatten()

    fig.suptitle('Distancia Coseno Pairwise por Género a lo Largo del Tiempo', fontsize=16, y=1.02)

    for i, genre in enumerate(target_genres):
        df_plot = unpivoted_df[unpivoted_df['Genre'] == genre]

        # Eliminar las comparaciones que ya no tienen datos después del filtro
        df_plot = df_plot.dropna(subset=['Cosine_Distance'])

        if df_plot.empty:
            continue

        sns.lineplot(
            data=df_plot,
            x='Year_Interval_Start',
            y='Cosine_Distance',
            hue='Compared_To', # El color es el género con el que se compara
            ax=axes[i],
            marker='o',
            linewidth=1.5
        )

        axes[i].set_title(f'Género Principal: {genre}', fontsize=12)
        axes[i].set_xlabel('Inicio del Intervalo', fontsize=10)
        axes[i].set_ylabel('Distancia Coseno', fontsize=10)
        axes[i].legend(title='Comparado con', fontsize=8)
        axes[i].grid(True)

    # Ocultar ejes vacíos
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()


if __name__ == "__main__":
    main()


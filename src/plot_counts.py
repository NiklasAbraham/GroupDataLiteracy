import json
import csv
import aiohttp
import asyncio
import matplotlib.pyplot as plt

from api.wikidata_handler import count_movies_by_year

def get_old_movie_counts(year: int):
    old_file_path = f"../all_data_with_embeddings/data_final/wikidata_movies_{year}.csv"
    try:
        with open(old_file_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            total_rows = sum(1 for _ in reader)
            return max(0, total_rows - 1)
    except FileNotFoundError:
        print(f"File not found: {old_file_path}")
        return 0
    except Exception as e:
        print(f"Error reading {old_file_path}: {e}")
        return 0

async def save_movie_counts_by_year(
    start_year=1950,
    end_year=2024
):
    years = list(range(start_year, end_year + 1))
    counts, old_counts = [], []
    async with aiohttp.ClientSession() as session:
        for year in years:
            count = await count_movies_by_year(session, year)
            counts.append(count)
            old_counts.append(get_old_movie_counts(year))
            print(f"Year {year}: {count} movies available in Wikidata vs {old_counts[-1]} in old dataset")
    
    output_path = f"movie_counts_{start_year}_{end_year}.csv"
    try:
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["year", "new_count", "old_count"])
            for year, new, old in zip(years, counts, old_counts):
                writer.writerow([year, new, old])
        print(f"Saved counts to {output_path}")
    except Exception as e:
        print(f"Error saving counts to {output_path}: {e}")


def plot_movie_counts_by_year(
    file_name: str = "movie_counts_1950_2024.csv",
):
    years, new_counts, old_counts = [], [], []
    try:
        with open(file_name, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                years.append(int(row['year']))
                new_counts.append(int(row['new_count']))
                old_counts.append(int(row['old_count']))
    except FileNotFoundError:
        print(f"File not found: {file_name}")
        return
    except Exception as e:
        print(f"Error reading {file_name}: {e}")
        return
    
    print("OLD COUNTS TOTAL:", sum(old_counts))
    print("NEW COUNTS TOTAL:", sum(new_counts))

    plt.figure(figsize=(12, 6))
    plt.plot(years, new_counts, label='Cleaned Data Movie Counts', marker='o')
    plt.plot(years, old_counts, label='Old Dataset Movie Counts', marker='x')
    plt.title('Movie Counts by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Movies')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()    

if __name__ == "__main__":
    # asyncio.run(save_movie_counts_by_year(2001, 2001))
    plot_movie_counts_by_year("movie_counts_1950_2024.csv")

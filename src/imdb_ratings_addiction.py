import pandas as pd
from pathlib import Path

BASE_DIR = Path.cwd().resolve()
DATA_FINAL_DIR = BASE_DIR / 'data' / 'data_final'
DATA_DIR = BASE_DIR / 'data'


# Load CSV file
csv_path = str(DATA_FINAL_DIR / 'final_dataset.csv')
df = pd.read_csv(csv_path)


df['tconst'] = df['imdb_id']

# Load IMDb ratings dataset
tsv_path = DATA_DIR / 'mock' / 'title_ratings.tsv'
imdb_ratings = pd.read_csv(tsv_path, sep='\t')

print(imdb_ratings.columns)

# Merge CSV with IMDb ratings
merged = df.merge(imdb_ratings, on='tconst', how='left')
merged = merged.drop(columns=['tconst'])

merged.rename(columns={
    'averageRating': 'imdb_rating',
    'numVotes': 'imdb_num_votes'
}, inplace=True)


# save to new CSV
merged.to_csv(str(DATA_FINAL_DIR / 'final_dataset_with_ratings.csv'), index=False)



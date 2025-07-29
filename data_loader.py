import pandas as pd
import os

def load_movie_data():
    """Load and process the movie datasets"""
    wiki_file = 'raw_wiki_movies.csv'
    imdb_file = 'raw_imdb_movies.csv'
    
    # Check if files exist
    if not os.path.exists(wiki_file) or not os.path.exists(imdb_file):
        raise FileNotFoundError("Movie data files not found. Please run the scraper first.")
    
    # Load datasets
    wiki_movies = pd.read_csv(wiki_file)
    imdb_movies = pd.read_csv(imdb_file)
    
    # Clean and process data as needed
    # For wiki_movies - ensure consistent column names
    wiki_movies.columns = [col.strip().lower() for col in wiki_movies.columns]
    # For imdb_movies - ensure consistent column names
    imdb_movies.columns = [col.strip().lower() for col in imdb_movies.columns]
    
    # Clean text data
    for df in [wiki_movies, imdb_movies]:
        if 'title' in df.columns:
            df['title'] = df['title'].str.strip()
        if 'year_released' in df.columns:
            df['year_released'] = df['year_released'].astype(str).str.strip()
    
    # Create a merged dataset for cross-referencing
    # Normalize titles for better matching
    wiki_movies['normalized_title'] = wiki_movies['title'].str.lower()
    imdb_movies['normalized_title'] = imdb_movies['title'].str.lower()
    
    # Find movies that appear in both lists by matching on normalized titles
    merged_movies = pd.merge(
        wiki_movies, 
        imdb_movies,
        on='normalized_title',
        how='inner',
        suffixes=('_gross', '_rated')
    )
    
    # Sort by combination of rating and rank
    if not merged_movies.empty and 'rating' in merged_movies.columns:
        merged_movies['combined_score'] = merged_movies['rating'].astype(float) * 10 - merged_movies['rank_gross'].astype(int)
        merged_movies = merged_movies.sort_values(by='combined_score', ascending=False)
    
    return {
        'highest_grossing': wiki_movies,
        'top_rated': imdb_movies,
        'cross_referenced': merged_movies
    }

if __name__ == "__main__":
    # Test data loading
    try:
        data = load_movie_data()
        print(f"Loaded {len(data['highest_grossing'])} highest grossing movies")
        print(f"Loaded {len(data['top_rated'])} top-rated movies")
        print(f"Found {len(data['cross_referenced'])} movies that appear in both lists")
    except Exception as e:
        print(f"Error loading data: {e}")
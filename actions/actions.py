from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
import pandas as pd
import os
import random
import numpy as np

# Make LangChain optional with graceful fallback
try:
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.llms import OpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available. Using fallback functionality.")

# Initialize LangChain only if available
if LANGCHAIN_AVAILABLE:
    try:
        # Get API key from environment variable - not hardcoded
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            llm = OpenAI(temperature=0.7)
            movie_description_template = PromptTemplate(
                input_variables=["movie_title", "year", "rating"],
                template="Write a brief, engaging description of the movie {movie_title} released in {year} with a rating of {rating}/10. Limit to 2-3 sentences."
            )
            movie_description_chain = LLMChain(llm=llm, prompt=movie_description_template)

            recommendation_template = PromptTemplate(
                input_variables=["movie_title"],
                template="Suggest 3 movies similar to {movie_title} that fans would enjoy. For each movie, give the title and a very brief reason why it's similar."
            )
            recommendation_chain = LLMChain(llm=llm, prompt=recommendation_template)
        else:
            LANGCHAIN_AVAILABLE = False
    except Exception as e:
        print(f"Error initializing LangChain: {e}")
        LANGCHAIN_AVAILABLE = False

# Load movie data
def load_movie_data():
    """Load and process the movie datasets"""
    wiki_file = 'raw_wiki_movies.csv'
    imdb_file = 'raw_imdb_movies.csv'
    
    # Check if files exist
    if not os.path.exists(wiki_file) or not os.path.exists(imdb_file):
        print("Movie data files not found. Please run the scraper first.")
        # Return empty dataframes instead of raising exception to maintain chatbot operation
        return {
            'highest_grossing': pd.DataFrame(columns=['rank', 'title', 'year_released']),
            'top_rated': pd.DataFrame(columns=['rank', 'title', 'year_released', 'rating']),
            'cross_referenced': pd.DataFrame()
        }
    
    try:
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
    
    except Exception as e:
        print(f"Error processing movie data: {e}")
        # Return empty dataframes to allow chatbot to continue operating
        return {
            'highest_grossing': pd.DataFrame(columns=['rank', 'title', 'year_released']),
            'top_rated': pd.DataFrame(columns=['rank', 'title', 'year_released', 'rating']),
            'cross_referenced': pd.DataFrame()
        }

# Helper functions for movie operations
def find_movie(title, movies_data):
    """Find a movie by title in our datasets"""
    results = []
    search_title = title.lower()
    
    # Search in highest grossing
    if 'highest_grossing' in movies_data and not movies_data['highest_grossing'].empty:
        # Ensure title column exists
        if 'title' in movies_data['highest_grossing'].columns:
            hg_results = movies_data['highest_grossing'][
                movies_data['highest_grossing']['title'].str.lower().str.contains(search_title)
            ]
            if not hg_results.empty:
                for _, row in hg_results.iterrows():
                    results.append({
                        'title': row['title'],
                        'year': row['year_released'],
                        'rank': row['rank'],
                        'rank_gross': row['rank'],
                        'genre': row.get('genre', 'N/A'),
                        'source': 'highest_grossing'
                    })
    
    # Search in top rated
    if 'top_rated' in movies_data and not movies_data['top_rated'].empty:
        # Ensure title column exists
        if 'title' in movies_data['top_rated'].columns:
            tr_results = movies_data['top_rated'][
                movies_data['top_rated']['title'].str.lower().str.contains(search_title)
            ]
            if not tr_results.empty:
                for _, row in tr_results.iterrows():
                    results.append({
                        'title': row['title'],
                        'year': row['year_released'],
                        'rank': row['rank'],
                        'rank_rated': row['rank'],
                        'rating': row.get('rating', 'N/A'),
                        'genre': row.get('genre', 'N/A'),
                        'source': 'top_rated'
                    })
    
    # Search in cross-referenced 
    if 'cross_referenced' in movies_data and not movies_data['cross_referenced'].empty:
        # Ensure title column exists
        if 'title_gross' in movies_data['cross_referenced'].columns:
            cr_results = movies_data['cross_referenced'][
                movies_data['cross_referenced']['title_gross'].str.lower().str.contains(search_title)
            ]
            if not cr_results.empty:
                for _, row in cr_results.iterrows():
                    results.append({
                        'title': row['title_gross'],
                        'year': row['year_released_gross'],
                        'rank': row['rank_gross'],
                        'rank_gross': row['rank_gross'],
                        'rank_rated': row['rank_rated'],
                        'rating': row.get('rating', 'N/A'),
                        'genre': row.get('genre', 'N/A'),
                        'source': 'cross_referenced'
                    })
    
    return results

def get_movie_recommendations_by_criteria(movies_data, criteria=None, limit=5):
    """Get movie recommendations based on different criteria"""
    recommendations = []
    
    if criteria is None:
        criteria = {}
    
    # Start with cross-referenced data for best results
    if 'cross_referenced' in movies_data and not movies_data['cross_referenced'].empty:
        df = movies_data['cross_referenced'].copy()
        
        # Apply filters based on criteria
        if 'year' in criteria and criteria['year'] and 'year_released_gross' in df.columns:
            df = df[df['year_released_gross'].astype(str).str.contains(str(criteria['year']))]
        
        if 'genre' in criteria and criteria['genre'] and 'genre' in df.columns:
            genre_filter = df['genre'].str.lower().str.contains(criteria['genre'].lower())
            df = df[genre_filter]
        
        # Sort based on combined criteria
        if 'sort' in criteria:
            if criteria['sort'] == 'rating' and 'rating' in df.columns:
                df = df.sort_values(by='rating', ascending=False)
            elif criteria['sort'] == 'year' and 'year_released_gross' in df.columns:
                df = df.sort_values(by='year_released_gross', ascending=False)
        else:
            # Default sort by combined score if appropriate columns exist
            if 'rating' in df.columns and 'rank_gross' in df.columns:
                df['combined_score'] = df['rating'].astype(float) * 10 - df['rank_gross'].astype(int)
                df = df.sort_values(by='combined_score', ascending=False)
        
        # Take top results
        top_results = df.head(limit).to_dict('records')
        for row in top_results:
            recommendations.append({
                'title': row['title_gross'],
                'year': row['year_released_gross'],
                'rank_gross': row['rank_gross'],
                'rank_rated': row['rank_rated'],
                'rating': row.get('rating', 'N/A'),
                'genre': row.get('genre', 'N/A')
            })
    
    # If we don't have enough recommendations, supplement from top_rated
    if len(recommendations) < limit and 'top_rated' in movies_data and not movies_data['top_rated'].empty:
        remaining = limit - len(recommendations)
        df = movies_data['top_rated'].copy()
        
        # Apply filters if columns exist
        if 'year' in criteria and criteria['year'] and 'year_released' in df.columns:
            df = df[df['year_released'].astype(str).str.contains(str(criteria['year']))]
        
        if 'genre' in criteria and criteria['genre'] and 'genre' in df.columns:
            df = df[df['genre'].str.lower().str.contains(criteria['genre'].lower())]
        
        # Sort by rating if column exists
        if 'rating' in df.columns:
            df = df.sort_values(by='rating', ascending=False)
        
        # Add to recommendations
        top_results = df.head(remaining).to_dict('records')
        for row in top_results:
            recommendations.append({
                'title': row['title'],
                'year': row['year_released'],
                'rank_rated': row['rank'],
                'rating': row.get('rating', 'N/A'),
                'genre': row.get('genre', 'N/A'),
                'source': 'top_rated'
            })
    
    # If still not enough, supplement from highest_grossing
    if len(recommendations) < limit and 'highest_grossing' in movies_data and not movies_data['highest_grossing'].empty:
        remaining = limit - len(recommendations)
        df = movies_data['highest_grossing'].copy()
        
        # Apply filters if columns exist
        if 'year' in criteria and criteria['year'] and 'year_released' in df.columns:
            df = df[df['year_released'].astype(str).str.contains(str(criteria['year']))]
        
        if 'genre' in criteria and criteria['genre'] and 'genre' in df.columns:
            df = df[df['genre'].str.lower().str.contains(criteria['genre'].lower())]
        
        # Sort by rank
        if 'rank' in df.columns:
            df = df.sort_values(by='rank')
        
        # Add to recommendations
        top_results = df.head(remaining).to_dict('records')
        for row in top_results:
            recommendations.append({
                'title': row['title'],
                'year': row['year_released'],
                'rank_gross': row['rank'],
                'genre': row.get('genre', 'N/A'),
                'source': 'highest_grossing'
            })
    
    return recommendations

class ActionSearchMovie(Action):
    def name(self) -> Text:
        return "action_search_movie"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        movie_title = next(tracker.get_latest_entity_values("movie"), None)
        if not movie_title:
            dispatcher.utter_message(text="I didn't catch which movie you're looking for. Can you please specify?")
            return []

        movies_data = load_movie_data()
        results = find_movie(movie_title, movies_data)
        
        if results:
            # Group results by title to combine information from different sources
            grouped_results = {}
            for movie in results:
                title = movie['title']
                if title not in grouped_results:
                    grouped_results[title] = {'sources': [], 'data': {}}
                
                grouped_results[title]['sources'].append(movie['source'])
                # Merge data, preferring more complete information
                for key, value in movie.items():
                    if key != 'source' and (key not in grouped_results[title]['data'] or grouped_results[title]['data'][key] == 'N/A'):
                        grouped_results[title]['data'][key] = value
            
            dispatcher.utter_message(text=f"I found information about movies matching '{movie_title}':")
            
            # Display the grouped results
            for title, info in list(grouped_results.items())[:3]:  # Limit to 3 results
                data = info['data']
                sources = info['sources']
                
                message = f"• {title} ({data.get('year', 'N/A')})"
                
                # Add ranking information based on sources
                ranks = []
                if 'highest_grossing' in sources:
                    ranks.append(f"#{data.get('rank_gross', data.get('rank', 'N/A'))} highest-grossing")
                if 'top_rated' in sources:
                    ranks.append(f"#{data.get('rank_rated', data.get('rank', 'N/A'))} top-rated with {data.get('rating', 'N/A')}/10 rating")
                if 'cross_referenced' in sources:
                    ranks.append("appears in both top-grossing and top-rated lists")
                
                if ranks:
                    message += f" - {', '.join(ranks)}"
                
                dispatcher.utter_message(text=message)
                
            # Set the slot for context
            return [SlotSet("movie", movie_title)]
        else:
            dispatcher.utter_message(text=f"I couldn't find information about '{movie_title}' in my database.")
            return []

class ActionGetHighestGrossing(Action):
    def name(self) -> Text:
        return "action_get_highest_grossing"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        movies_data = load_movie_data()
        highest_grossing = movies_data.get('highest_grossing', pd.DataFrame())
        
        if highest_grossing.empty:
            dispatcher.utter_message(text="I'm sorry, I don't have information about highest grossing movies right now.")
            return []
        
        top_5 = highest_grossing.head(5)
        
        message = "Here are the top 5 highest-grossing films of all time:\n\n"
        for _, movie in top_5.iterrows():
            message += f"• #{movie['rank']} - {movie['title']} ({movie['year_released']})\n"
        
        dispatcher.utter_message(text=message)
        
        # Set context for follow-up
        return [SlotSet("search_context", "highest_grossing")]

class ActionGetTopRated(Action):
    def name(self) -> Text:
        return "action_get_top_rated"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        movies_data = load_movie_data()
        top_rated = movies_data.get('top_rated', pd.DataFrame())
        
        if top_rated.empty:
            dispatcher.utter_message(text="I'm sorry, I don't have information about top-rated movies right now.")
            return []
        
        top_5 = top_rated.head(5)
        
        message = "Here are the top 5 highest-rated movies according to IMDb:\n\n"
        for _, movie in top_5.iterrows():
            rating = movie.get('rating', 'N/A')
            message += f"• #{movie['rank']} - {movie['title']} ({movie['year_released']}) - Rating: {rating}/10\n"
        
        dispatcher.utter_message(text=message)
        
        # Set context for follow-up
        return [SlotSet("search_context", "top_rated")]

class ActionGetMovieDetails(Action):
    def name(self) -> Text:
        return "action_get_movie_details"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        movie_title = next(tracker.get_latest_entity_values("movie"), None)
        if not movie_title:
            movie_title = tracker.get_slot("movie")
            
        if not movie_title:
            dispatcher.utter_message(text="I didn't catch which movie you're asking about. Can you please specify?")
            return []
        
        movies_data = load_movie_data()
        results = find_movie(movie_title, movies_data)
        
        if results:
            # Take the best match - fixed: properly handle results as a list
            movie = results[0] if results else None
            
            if movie:
                # Use LangChain to generate a nice description if available
                if LANGCHAIN_AVAILABLE and 'movie_description_chain' in globals():
                    try:
                        year = movie.get('year', 'unknown year')
                        rating = movie.get('rating', '8.0')  # Default if not available
                        
                        description = movie_description_chain.run(
                            movie_title=movie['title'],
                            year=year,
                            rating=rating
                        )
                        
                        # Craft the response message
                        if movie['source'] == 'highest_grossing':
                            message = f"{movie['title']} ({year}): Ranked #{movie.get('rank', 'N/A')} in highest-grossing films of all time.\n\n{description}"
                        else:
                            message = f"{movie['title']} ({year}): Ranked #{movie.get('rank', 'N/A')} in IMDb's top-rated movies with a rating of {rating}/10.\n\n{description}"
                            
                        dispatcher.utter_message(text=message)
                    except Exception as e:
                        # Fallback if LangChain fails
                        basic_message = self._create_basic_movie_details(movie)
                        dispatcher.utter_message(text=basic_message)
                else:
                    # Fallback to basic details without LangChain
                    basic_message = self._create_basic_movie_details(movie)
                    dispatcher.utter_message(text=basic_message)
                    
                return [SlotSet("movie", movie_title)]
            else:
                dispatcher.utter_message(text=f"I found results for '{movie_title}', but couldn't retrieve detailed information.")
                return []
        else:
            dispatcher.utter_message(text=f"I couldn't find detailed information about '{movie_title}' in my database.")
            return []
    
    def _create_basic_movie_details(self, movie):
        """Create a basic movie details message without using LangChain"""
        title = movie.get('title', 'Unknown')
        year = movie.get('year', 'Unknown year')
        
        details = []
        if movie.get('source') == 'highest_grossing':
            rank = movie.get('rank', movie.get('rank_gross', 'N/A'))
            details.append(f"Ranked #{rank} in highest-grossing films")
        
        if movie.get('source') == 'top_rated' or 'rating' in movie:
            rank = movie.get('rank', movie.get('rank_rated', 'N/A'))
            rating = movie.get('rating', 'N/A')
            details.append(f"Ranked #{rank} in top-rated movies with a rating of {rating}/10")
        
        if movie.get('source') == 'cross_referenced':
            gross_rank = movie.get('rank_gross', 'N/A')
            rated_rank = movie.get('rank_rated', 'N/A')
            rating = movie.get('rating', 'N/A')
            details.append(f"Ranked #{gross_rank} in highest-grossing and #{rated_rank} in top-rated films with a rating of {rating}/10")
        
        if not details:
            return f"{title} ({year}) - No additional details available."
        
        return f"{title} ({year}) - {'; '.join(details)}."

class ActionRecommendMovies(Action):
    def name(self) -> Text:
        return "action_recommend_movies"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Get preferred movie if mentioned
        movie_title = next(tracker.get_latest_entity_values("movie"), None)
        if not movie_title:
            movie_title = tracker.get_slot("movie")
        
        # Get genre if mentioned
        genre = next(tracker.get_latest_entity_values("genre"), None)
        
        # Get year if mentioned
        year = next(tracker.get_latest_entity_values("year"), None)
        
        movies_data = load_movie_data()
        
        criteria = {}
        if genre:
            criteria['genre'] = genre
        if year:
            criteria['year'] = year
            
        if movie_title and LANGCHAIN_AVAILABLE and 'recommendation_chain' in globals():
            # If the user mentioned a movie, use LangChain to generate recommendations
            try:
                recommendations = recommendation_chain.run(movie_title=movie_title)
                
                message = f"If you enjoyed {movie_title}, you might like these similar movies:\n\n{recommendations}"
                dispatcher.utter_message(text=message)
                return []
            except Exception as e:
                # Fallback to data-driven recommendations if LangChain fails
                # First try to find the movie in our database
                movie_results = find_movie(movie_title, movies_data)
                if movie_results:
                    # Extract genre from the found movie
                    found_movie = movie_results[0]
                    if 'genre' in found_movie and found_movie['genre'] != 'N/A':
                        criteria['genre'] = found_movie['genre']
        
        # Get recommendations based on criteria
        recommended_movies = get_movie_recommendations_by_criteria(movies_data, criteria)
        
        if recommended_movies:
            # Build a context-aware message
            if genre and year:
                message = f"Here are some excellent {genre} movies from {year} you might enjoy:\n\n"
            elif genre:
                message = f"Here are some excellent {genre} movies you might enjoy:\n\n"
            elif year:
                message = f"Here are some excellent movies from {year} you might enjoy:\n\n"
            elif movie_title:
                message = f"Based on your interest in {movie_title}, here are some movies you might enjoy:\n\n"
            else:
                message = "Here are some excellent movies you might enjoy:\n\n"
            
            for movie in recommended_movies:
                title = movie['title']
                year = movie.get('year', 'N/A')
                rating = movie.get('rating', 'N/A')
                
                movie_info = f"• {title} ({year})"
                
                # Add ratings if available
                if rating != 'N/A':
                    movie_info += f" - Rating: {rating}/10"
                
                # Add ranks if available
                ranks = []
                if 'rank_gross' in movie:
                    ranks.append(f"#{movie['rank_gross']} highest-grossing")
                if 'rank_rated' in movie:
                    ranks.append(f"#{movie['rank_rated']} top-rated")
                
                if ranks:
                    movie_info += f" ({', '.join(ranks)})"
                
                message += movie_info + "\n"
            
            dispatcher.utter_message(text=message)
        else:
            dispatcher.utter_message(text="I'm sorry, I don't have movie recommendations available that match your criteria.")
        
        return []

class ActionGetSimilarMovies(Action):
    def name(self) -> Text:
        return "action_get_similar_movies"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        movie_title = next(tracker.get_latest_entity_values("movie"), None)
        if not movie_title:
            dispatcher.utter_message(text="Which movie would you like to find similar films to?")
            return []
        
        # Use LangChain to generate similar movie suggestions if available
        if LANGCHAIN_AVAILABLE and 'recommendation_chain' in globals():
            try:
                recommendations = recommendation_chain.run(movie_title=movie_title)
                
                message = f"If you enjoyed {movie_title}, you might like these similar movies:\n\n{recommendations}"
                dispatcher.utter_message(text=message)
                return [SlotSet("movie", movie_title)]
            except Exception as e:
                # Fall through to alternate method if LangChain fails
                pass
        
        # Fallback: use data-driven similarity
        movies_data = load_movie_data()
        movie_results = find_movie(movie_title, movies_data)
        
        if movie_results:
            found_movie = movie_results[0]
            criteria = {}
            
            # Extract criteria for similarity
            if 'genre' in found_movie and found_movie['genre'] != 'N/A':
                criteria['genre'] = found_movie['genre']
            
            # Get similar movies excluding the original
            similar_movies = get_movie_recommendations_by_criteria(movies_data, criteria, limit=5)
            similar_movies = [m for m in similar_movies if m['title'].lower() != movie_title.lower()]
            
            if similar_movies:
                message = f"If you enjoyed {movie_title}, you might like these similar movies:\n\n"
                
                for movie in similar_movies[:3]:  # Limit to top 3
                    title = movie['title']
                    year = movie.get('year', 'N/A')
                    rating = movie.get('rating', 'N/A')
                    
                    movie_info = f"• {title} ({year})"
                    
                    # Add ratings if available
                    if rating != 'N/A':
                        movie_info += f" - Rating: {rating}/10"
                    
                    message += movie_info + "\n"
                
                dispatcher.utter_message(text=message)
            else:
                dispatcher.utter_message(text=f"I'm having trouble finding movies similar to {movie_title} right now.")
        else:
            dispatcher.utter_message(text=f"I don't have information about {movie_title} to find similar movies.")
        
        return [SlotSet("movie", movie_title)]

class ActionGetMoviesByYear(Action):
    def name(self) -> Text:
        return "action_get_movies_by_year"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        year = next(tracker.get_latest_entity_values("year"), None)
        if not year:
            dispatcher.utter_message(text="Which year are you interested in?")
            return []
        
        movies_data = load_movie_data()
        results = []
        
        # Search in highest grossing
        if 'highest_grossing' in movies_data and not movies_data['highest_grossing'].empty:
            if 'year_released' in movies_data['highest_grossing'].columns:
                hg_results = movies_data['highest_grossing'][
                    movies_data['highest_grossing']['year_released'].astype(str).str.contains(str(year))
                ]
                if not hg_results.empty:
                    for _, row in hg_results.iterrows():
                        results.append({
                            'title': row['title'],
                            'rank': row['rank'],
                            'source': 'highest_grossing'
                        })
        
        # Search in top rated
        if 'top_rated' in movies_data and not movies_data['top_rated'].empty:
            if 'year_released' in movies_data['top_rated'].columns:
                tr_results = movies_data['top_rated'][
                    movies_data['top_rated']['year_released'].astype(str).str.contains(str(year))
                ]
                if not tr_results.empty:
                    for _, row in tr_results.iterrows():
                        results.append({
                            'title': row['title'],
                            'rank': row['rank'],
                            'rating': row.get('rating', 'N/A'),
                            'source': 'top_rated'
                        })
        
        if results:
            # Limit to maximum 5 results
            results = results[:5]
            
            message = f"Here are some notable movies from {year}:\n\n"
            for movie in results:
                source_text = "highest-grossing" if movie['source'] == 'highest_grossing' else "top-rated"
                rating_info = f" - Rating: {movie['rating']}/10" if 'rating' in movie else ""
                
                message += f"• {movie['title']} (#{movie['rank']} {source_text}){rating_info}\n"
            
            dispatcher.utter_message(text=message)
        else:
            dispatcher.utter_message(text=f"I couldn't find any movies from {year} in my database.")
        
        return [SlotSet("year", year)]

class ActionGetTopRatedAndGrossing(Action):
    def name(self) -> Text:
        return "action_get_top_rated_and_grossing"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        movies_data = load_movie_data()
        cross_referenced = movies_data.get('cross_referenced', pd.DataFrame())
        
        if cross_referenced.empty:
            dispatcher.utter_message(text="I'm sorry, I couldn't find movies that are both top-rated and highest-grossing.")
            return []
        
        # Take top 5 movies from cross-referenced data
        top_5 = cross_referenced.head(5)
        
        message = "Here are movies that are both critically acclaimed and commercially successful:\n\n"
        
        for _, movie in top_5.iterrows():
            # Check if required columns exist to avoid KeyError
            if 'title_gross' in movie and 'year_released_gross' in movie and 'rank_gross' in movie and 'rank_rated' in movie:
                title = movie['title_gross']  # Use the title from gross list (they should match anyway)
                year = movie['year_released_gross']
                rating = movie.get('rating', 'N/A')
                gross_rank = movie['rank_gross']
                rated_rank = movie['rank_rated']
                
                message += f"• {title} ({year}) - #{gross_rank} highest-grossing, #{rated_rank} top-rated with {rating}/10 rating\n"
        
        dispatcher.utter_message(text=message)
        
        # Set context for follow-up
        return [SlotSet("search_context", "cross_referenced")]

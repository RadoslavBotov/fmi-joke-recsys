import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from src.metrics import PredictionTuple, get_mae, get_mrr, get_rmse
from src.utils import get_mean_for_users
from tqdm import tqdm


def vectorize_df(df: pd.DataFrame, column: str, use_tfidf=True) -> np.ndarray:
    """
    Transforms text data into numerical vectors using TF-IDF or Bag of Words.
    """
    if (use_tfidf is True):
        print('>> TF-IDF')
        vect = TfidfVectorizer(lowercase=True, stop_words='english')
    else:
        print('>> BOW')
        vect = CountVectorizer(lowercase=True, stop_words='english')
    return vect.fit_transform(df[column])

def get_similarity_scores(joke_id: int, user_rated_joke_ids: list[int], cos_sim: np.ndarray) -> np.ndarray:
    """
    Retrieves cosine similarity scores between a target joke and all jokes previously rated by a user.
    """
    joke_vector = cos_sim[joke_id]
    user_rated_joke_vectors = cos_sim[user_rated_joke_ids]
    # Calculate similarity between the target joke and the user's history
    return cosine_similarity(joke_vector.reshape(1, -1), user_rated_joke_vectors).flatten()

def weight_joke_prediction(similarity_scores: np.ndarray, user_rated_joke_ratings: np.ndarray) -> float:
    """
    Calculates a predicted rating based on a weighted average of similar items.
    """
    weighted_sum = np.dot(similarity_scores, user_rated_joke_ratings)
    sum_of_weights = np.sum(similarity_scores)

    # Fallback to mean rating if no similar jokes are found in the training data
    if sum_of_weights == 0:
        return float(user_rated_joke_ratings.mean())

    return float(weighted_sum / sum_of_weights)

def predict_rating(user_id: int, joke_id: int, train_df: pd.DataFrame, cos_sim: np.ndarray) -> float:
    """
    Predicts the rating for a specific user/joke pair.
    """
    # Fallback if joke was never seen in training
    if joke_id not in train_df['joke_id']:
        return train_df['rating'].mean()

    user_rated_jokes = train_df[train_df['user_id'] == user_id]

    # Fallback if user has no ratings in training data
    if user_rated_jokes.empty:
        return train_df[train_df['joke_id'] == joke_id]['rating'].mean()

    similarity_scores = get_similarity_scores(joke_id, user_rated_jokes['joke_id'].to_list(), cos_sim)
    return weight_joke_prediction(similarity_scores, user_rated_jokes['rating'].values)

def predict_ratings(train: pd.DataFrame, test: pd.DataFrame, cos_sim: np.ndarray) -> List[Tuple[int, float, float]]:
    """
    Generates predictions for all entries in the test set.
    """
    predictions = []
    for _, user_id, joke_id, rating in tqdm(test.itertuples(), total=len(test)):
        pred = predict_rating(user_id, joke_id, train, cos_sim)
        predictions.append((user_id, rating, pred))
    return predictions

def translate_tfidf_predictions(predictions: List[Tuple[int, float, float]], user_means: Dict[int, float]) -> List[PredictionTuple]:
    """
    Adds user mean statistics to prediction results for metrics evaluation.
    """
    return [
        (user_id, actual, pred, user_means[user_id])
        for user_id, actual, pred in predictions
    ]

def tfidf_calcs(args_dict: Dict[str, str|int]) -> None:
    """
    Orchestrates the Content-Based recommendation pipeline using TF-IDF/BOW embeddings.
    """
    # Load data
    jester_jokes_df = pd.read_csv(os.path.join('data', 'jester_jokes.csv')) # always the same jokes
    jester_data_svd_df = pd.read_csv(args_dict['data'])

    # Generate embeddings based on user-selected mode (TF-IDF vs Bag-of-Words)
    X = vectorize_df(jester_jokes_df, 'JokeText', use_tfidf=True if args_dict['tfidf_mode'] == 'tf' else False) # (100, 1378)

    # Compute joke-to-joke similarity matrix
    cos_sim = cosine_similarity(X, X) # (100, 100)
    
    # Split data while maintaining user stratification
    train, test = train_test_split(
        jester_data_svd_df,
        test_size=0.2,
        random_state=42,
        stratify=jester_data_svd_df['user_id']
    )
    user_means = get_mean_for_users(train)

    # Predictions: list[(user_id, true, predicted)]
    predictions = predict_ratings(train, test, cos_sim)

    # Metrics
    print(f'> Metrics for \'{args_dict['data']}\'')
    print(f'RMSE: {get_rmse(predictions)}')
    print(f'MAE: {get_mae(predictions)}')

    predictions = translate_tfidf_predictions(predictions, user_means)
    print(f'MRR: {get_mrr(predictions)}')

# TF-IDF
# Time: 47m 18s
# > Metrics for 'data_svd\jester_data_1_svd.csv'
# MRSE: 4.7943
# MAE:  3.8091
# MRR:  0.7148

# > Metrics for 'data_svd\jester_data_2_svd.csv'
# RMSE: 5.1830
# MAE:  4.0869
# MRR:  0.6562

# > Metrics for 'data_svd\jester_data_comb_svd.csv'
# RMSE: 4.9288
# MAE:  3.9316
# MRR:  0.6848

# BOW
# > Metrics for 'data_svd\jester_data_1_svd.csv'
# RMSE: 4.7798
# MAE:  3.8019
# MRR:  0.7116

# > Metrics for 'data_svd\jester_data_2_svd.csv'
# RMSE: 5.1509
# MAE:  4.0691
# MRRL  0.6531

# > Metrics for 'data_svd\jester_data_comb_svd.csv'
# RMSE: 4.9107
# MAE:  3.9229
# MRR:  0.6846

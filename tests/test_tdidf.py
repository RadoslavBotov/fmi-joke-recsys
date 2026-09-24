import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from src.tfidf import predict_ratings, translate_tfidf_predictions, weight_joke_prediction, get_similarity_scores, predict_rating, vectorize_df


class TestVectorizeDf(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'dummy': ['1', '2', '3'],
            'text': ['the cat', 'sat on', 'the cat']
        })

    def test_tfidf_vectorization(self):
        # Act:
        result = vectorize_df(self.df, 'text', use_tfidf=True)
        
        # Assert:
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result.data[0], 1)
        self.assertEqual(result.data[1], 1)
        self.assertEqual(result.data[2], 1)
        self.assertEqual(result.dtype, float)
        self.assertEqual(type(result), scipy.sparse._csr.csr_matrix)

    def test_bow_vectorization(self):
        # Act:
        result = vectorize_df(self.df, 'text', use_tfidf=False)
        
        # Assert:
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result.data[0], 1)
        self.assertEqual(result.data[1], 1)
        self.assertEqual(result.data[2], 1)
        self.assertEqual(result.dtype, int)
        self.assertEqual(type(result), scipy.sparse._csr.csr_matrix)


class TestGetSimilarityScores(unittest.TestCase):
    def setUp(self):
        self.cos_sim = np.array([
            # j0   j1   j2   j3
            [1.0, 0.8, 0.2, 0.1], # j0
            [0.8, 1.0, 0.3, 0.0], # j1
            [0.2, 0.3, 1.0, 0.5], # j2
            [0.1, 0.0, 0.5, 1.0]  # j3
        ])

    def test_similarity_score_values(self):
        # Arrange:
        target_joke = 0
        user_rated = [1, 2]
        expected = cosine_similarity(self.cos_sim[target_joke].reshape(1, -1), self.cos_sim[user_rated]).flatten()
        
        # Act
        scores = get_similarity_scores(target_joke, user_rated, self.cos_sim)
        
        # Assert:
        self.assertEqual(len(scores), 2)
        self.assertEqual(type(scores), np.ndarray)
        self.assertListEqual(list(scores), list(expected))

    def test_single_rated_joke(self):
        # Arrange:
        target_joke = 3
        user_rated = [2]
        
        # Act
        scores = get_similarity_scores(target_joke, user_rated, self.cos_sim)
        
        # Assert
        # j3: 0.1, 0.0, 0.5, 1.0 # norm = 1.1224
        # j2: 0.2, 0.3, 1.0, 0.5 # norm = 1.1747
        # 0.02 + 0 + 0.5 + 0.5 = 1.02
        # 1.02 / (1.1224 * 1.1747) = 0.77352678
        self.assertEqual(len(scores), 1)
        self.assertEqual(type(scores), np.ndarray)
        self.assertAlmostEqual(scores[0], 0.77352678)


class TestGetPredictedRating(unittest.TestCase):

    def test_weighted_average_calculation(self):
        # Arrange:
        # Similarity: 0.8, 0.2
        # Ratings: 3, 4
        # Expected: (0.8*3 + 0.2*4) / (0.8 + 0.2) = 3.2 / 1 = 3.2
        sim_scores = np.array([0.8, 0.2])
        ratings = np.array([3, 4])

        # Act
        prediction = weight_joke_prediction(sim_scores, ratings)

        # Assert
        self.assertEqual(prediction, 3.2)

    def test_fallback_to_mean_when_no_similarity(self):
        # Arrange:
        sim_scores = np.array([0, 0])
        ratings = np.array([3, 4])

        # Act
        prediction = weight_joke_prediction(sim_scores, ratings)

        # Assert:
        self.assertEqual(prediction, 3.5)

    def test_single_item_prediction(self):
        # Arrange
        sim_scores = np.array([0.5])
        ratings = np.array([10])
        # (10 * 0.5) / 0.5 = 10
        
        # Act
        prediction = weight_joke_prediction(sim_scores, ratings)
        
        # Assert
        self.assertEqual(prediction, 10)


class TestPredictRating(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.DataFrame({
            'user_id': [1, 1, 2],
            'joke_id': [0, 1, 1],
            'rating': [5, 3, 4]
        })
        self.cos_sim = np.array([
            [1.0, 0.5, 0.1],
            [0.5, 1.0, 0.2],
            [0.1, 0.2, 1.0]
        ])

    def test_predict_existing_user_and_joke(self):
        # Predicts rating user 1 would give to joke 1 (actual = 3)
        # Gets jokes rated by user 1 => jokes 0 and 1
        # 

        # Act
        prediction = predict_rating(1, 1, self.train_df, self.cos_sim)
        
        # Assert
        self.assertIsInstance(prediction, float)
        self.assertAlmostEqual(prediction, 3.8889230662)

    def test_predict_unseen_joke(self):
        # Joke 99 doesn't exist it dataset
        # Should return mean for user 1: (5+3)/2 = 4

        # Act
        prediction = predict_rating(1, 99, self.train_df, self.cos_sim)
        
        # Assert
        self.assertEqual(prediction, 4)

    def test_predict_user_with_no_history(self):
        # User 99 has no ratings.
        # Should return mean of the target (joke 1): (3+4)/2 = 3.5

        # Act
        prediction = predict_rating(99, 1, self.train_df, self.cos_sim)

        # Assert
        self.assertEqual(prediction, 3.5)


class TestPredictRatings(unittest.TestCase):

    @patch('src.tfidf.predict_rating')
    def test_predict_ratings_iteration(self, mock_predict_rating):
        # Arrange:
        test_df = pd.DataFrame({
            'user_id': [1, 2, 3],
            'joke_id': [0, 1, 2],
            'rating': [5, 4, 3]
        })
        train_df = MagicMock()
        cos_sim = MagicMock()

        mock_predict_rating.return_value = 3.5

        # Act
        results = predict_ratings(train_df, test_df, cos_sim)

        # Assert
        self.assertEqual(mock_predict_rating.call_count, 3)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], (1, 5, 3.5))
        self.assertEqual(results[1], (2, 4, 3.5))
        self.assertEqual(results[2], (3, 3, 3.5))


class TestTranslateTfidfPredictions(unittest.TestCase):
    
    def test_translate_empty_input(self):
        # Arrange
        predictions = []
        user_means = {}
        
        # Act
        result = translate_tfidf_predictions(predictions, user_means)
        
        # Assert
        self.assertEqual(result, [])

    def test_translate_standard(self):
        # Arrange:
        predictions = [
            (1, 5, 3),
            (1, 4, 2),
            (2, 4, 4),
        ]
        user_means = {1: 2, 2: 3.5}
        
        # Act
        result = translate_tfidf_predictions(predictions, user_means)
        
        # Assert: Expect (user_id, actual, pred, user_mean)
        self.assertEqual(result, [(1, 5, 3, 2), (1, 4, 2, 2), (2, 4, 4, 3.5)])

    def test_translate_missing_user_mean_raises_keyerror(self):
        # Arrange: User 1 is missing a mean
        predictions = [(1, 5.0, 3.5)]
        user_means = {}
        
        # Act and Assert
        with self.assertRaises(KeyError):
            translate_tfidf_predictions(predictions, user_means)

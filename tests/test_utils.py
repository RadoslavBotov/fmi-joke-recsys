import unittest
import pandas as pd

from src.utils import get_mean_for_users, get_predictions_per_user


class TestUserMeanCalculations(unittest.TestCase):
    def test_get_mean_for_users_correctness(self):
        # Arrange
        df = pd.DataFrame({
            'user_id': [1, 2, 2, 2],
            'item_id': [1, 1, 2, 3],
            'rating': [5, 3, 4, 2]
        })
        # User 1: (5)/1 = 5.0
        # User 2: (3+4+2)/3 = 3.0

        # Act:
        result = get_mean_for_users(df)

        # Assert:
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[1], 5.0)
        self.assertAlmostEqual(result[2], 3.0)

    def test_get_mean_for_users_single_entry(self):
        # Arrange:
        df = pd.DataFrame({
            'user_id': [1, 1, 1],
            'item_id': [1, 2, 3],
            'rating': [5, 3, 2]
        })
        # User 1: (5+3+2)/3 = 3.333333333333

        # Act
        result = get_mean_for_users(df)

        # Assert
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[1], 3.333333333333)


class TestPredictionsPerUser(unittest.TestCase):
    def test_get_predictions_per_user_empty(self):
        # Arrange:
        predictions = []

        # Act
        result = get_predictions_per_user(predictions)

        # Assert
        self.assertEqual(result, {})

    def test_get_predictions_per_user_grouping(self):
        # Arrange: (user_id, actual_rating, predicted_rating, user_mean)
        predictions = [
            (1, 4, 3, 4),
            (1, 2, 2, 4),
            (2, 5, 4, 2)
        ]

        # Act: Execute the function
        result = get_predictions_per_user(predictions)

        # Assert: user 1
        self.assertIn(1, result)
        self.assertEqual(result[1][0], 4)                   # Verify user_mean
        self.assertEqual(len(result[1][1]), 2)              # Verify number of predictions
        self.assertEqual(result[1][1], [(4, 3), (2, 2)])    # Verify specific pair

        # Assert: user 2
        self.assertIn(2, result)
        self.assertEqual(result[2][0], 2)           # Verify user_mean
        self.assertEqual(len(result[2][1]), 1)      # Verify number of predictions
        self.assertEqual(result[2][1], [(5, 4)])    # Verify specific pair

import unittest
from unittest.mock import MagicMock, patch

from surprise import Prediction
from src.svd import calculate_algo, translate_surprise_predictions

class TestTranslateSurprisePredictions(unittest.TestCase):

    def test_empty_predictions(self):
        # Arrange
        predictions = []
        user_means = {}

        # Act
        result = translate_surprise_predictions(predictions, user_means)

        # Assert
        self.assertEqual(result, [])

    def test_multiple_predictions_same_user(self):
        # Arrange:
        # Prediction(uid, iid, r_ui, est, details)
        predictions = [
            Prediction(uid=1, iid=101, r_ui=4, est=3, details={}),
            Prediction(uid=1, iid=102, r_ui=5, est=4, details={})
        ]
        user_means = {1: 3}

        # Act
        result = translate_surprise_predictions(predictions, user_means)

        # Assert:
        # (user_id, real_rating, estimate, user_mean)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], (1, 4, 3, 3)) # user_1
        self.assertEqual(result[1], (1, 5, 4, 3)) # user_2

    def test_multiple_predictions_two_user(self):
        # Arrange: 
        # Prediction(uid, iid, r_ui, est, details)
        predictions = [
            Prediction(uid=1, iid=101, r_ui=4, est=3, details={}),
            Prediction(uid=1, iid=102, r_ui=5, est=4, details={}),
            Prediction(uid=2, iid=103, r_ui=1, est=3, details={}),
            Prediction(uid=2, iid=104, r_ui=4, est=2, details={})
        ]
        user_means = {1: 3, 2: 2}

        # Act
        result = translate_surprise_predictions(predictions, user_means)

        # Assert:
        # (user_id, real_rating, estimate, user_mean)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], (1, 4, 3, 3)) # user_1
        self.assertEqual(result[1], (1, 5, 4, 3)) # user_1
        self.assertEqual(result[2], (2, 1, 3, 2)) # user_2
        self.assertEqual(result[3], (2, 4, 2, 2)) # user_2

    def test_missing_user_in_mean_raises_keyerror(self):
        # Arrange:
        predictions = [
            Prediction(uid=99, iid=101, r_ui=4, est=3, details={})
        ]
        user_means = {1: 3}

        # Act & Assert
        with self.assertRaises(KeyError):
            translate_surprise_predictions(predictions, user_means)


class TestCalculateAlgo(unittest.TestCase):

    # @patch('src.svd.accuracy')
    # @patch('src.svd.get_mrr')
    # @patch('src.svd.translate_surprise_predictions')
    def test_calculate_algo_pipeline(self):
        # Arrange
        mock_algo = MagicMock()
        mock_train = MagicMock()
        mock_test = MagicMock()
        user_means = {1: 3}
        mock_algo.test.return_value = [
            Prediction(uid=1, iid=101, r_ui=4, est=3, details={}),
            Prediction(uid=1, iid=101, r_ui=2, est=3, details={})
        ]

        # Act
        rmse, mae, mrr = calculate_algo(mock_algo, mock_train, mock_test, user_means)
        
        # Assert
        mock_algo.fit.assert_called_once_with(mock_train)
        mock_algo.test.assert_called_once_with(mock_test)
        self.assertAlmostEqual(rmse, 1, msg=f"{rmse}") # sqrt[((4-3)^2 + (2-3)^2)/2] = sqrt[(1^2 + 1^2)/2] = sqrt(2/2) = sqrt(1) = 1
        self.assertAlmostEqual(mae, 1, msg=f"{mae}") # (|4-3| + |2-3|)/2 = (1+1)/2 = 2/2 = 1
        self.assertAlmostEqual(mrr, 1, msg=f"{mrr}") # 1st relevant with only 1 user => 1/1 = 1



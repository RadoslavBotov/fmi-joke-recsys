import unittest

import numpy as np
from src.metrics import get_mae, get_mrr, get_rmse

class TestGetRMSE(unittest.TestCase):

    def test_get_rmse_values(self):
        # Arrange: 
        # (user_id, actual, pred, user_mean)
        predictions = [
            (1, 4, 3, 3),
            (1, 2, 3, 3),
            (2, -3, -1, 2)
        ]
        # Errors: (4-3)^2=1, (2-3)^2=1, (-3+1)^2=4 
        # MSE = (1 + 1 + 4) / 3 = 2
        # RMSE = sqrt(2)

        # Act
        result = get_rmse(predictions)

        # Assert
        self.assertAlmostEqual(result, np.sqrt(2))

    def test_get_rmse_perfect_prediction(self):
        # Arrange:
        predictions = [
            (1, 5, 5, 4),
            (1, 4, 4, 4),
            (2, 3, 3, 3),
            (3, 1, 1, 2)
        ]
        # Errors: (5-5)^2=0, (4-4)^2=0, (3-3)^2=0 
        # MSE = (0 + 0 + 0) / 3 = 0
        # RMSE = sqrt(0) = 0

        # Act
        result = get_rmse(predictions)

        # Assert
        self.assertEqual(result, 0)


class TestGetMAE(unittest.TestCase):

    def test_get_mae_values(self):
        # Arrange: 
        # (user_id, actual, pred, user_mean)
        predictions = [
            (1, 4, 3, 3),
            (1, 2, 3, 3),
            (2, -3, -1, 2)
        ]
        # Errors: |4-3|=1, |2-3|=1, |-3+1|=2 
        # MAE = (1 + 1 + 2) / 3 = 1.3333333333

        # Act
        result = get_mae(predictions)

        # Assert
        self.assertAlmostEqual(result, 1.3333333333)

    def test_get_mae_zero_error(self):
        # Arrange:
        predictions = [
            (1, 5, 5, 4),
            (1, 4, 4, 4),
            (2, 3, 3, 3),
            (3, 1, 1, 2)
        ]
        # Errors: |5-5|=0, |4-4|=0, |3-3|=0 
        # MAE = (0 + 0 + 0) / 3 = 0

        # Act
        result = get_mae(predictions)

        # Assert
        self.assertEqual(result, 0)


class TestGetMRR(unittest.TestCase):

    def test_get_mrr_perfect_rank_perfect(self):
        # Arrange:
        predictions = [
            (1, 5, 5, 3),
            (1, 4, 4, 3),
            (2, 3, 3, 2),
            (3, 1, 1, 1)
        ]
        # predictions is transformed internally to
        # dict {
        #     1: (3, [(5, 5), (4, 4)]), # 1st relevant
        #     2: (2, [(3, 3)])          # 1st relevant
        #     3: (1, [(3, 1)])          # 1st relevant
        # }

        # Act
        result = get_mrr(predictions)
        
        # Assert: 1/1 = 1.0
        self.assertEqual(result, 1.0)

    def test_get_mrr_perfect_rank_positive(self):
        # Arrange:
        predictions = [
            (1, 4, 3, 1),
            (1, 2, 3, 1),
            (2, 3, 1, 1)
        ]
        # predictions is transformed internally to
        # dict {
        #     1: (1, [(4, 3), (2, 3)]), # 1st relevant
        #     2: (1, [(3, 1)])          # 1st relevant
        # }

        # Act
        result = get_mrr(predictions)
        
        # Assert:
        self.assertEqual(result, 1)

    def test_get_mrr_perfect_rank_negative(self):
        # Arrange: 
        predictions = [
            (1, -2, -3, -4),
            (1, -2, -4, -4),
            (2, -3, -1, -3)
        ]
        # predictions is transformed internally to
        # dict {
        #     1: (-4, [(-2, -3), (-2, -4)]), # no relevant
        #     2: (-3, [(-3, -1)])            # no relevant
        # }

        # Act
        result = get_mrr(predictions)
        
        # Assert:
        self.assertEqual(result, 1)

    def test_get_mrr_second_rank(self):
        # Arrange: Relevant item at rank 2
        predictions = [
            (1, 2, 4, 3), # actual not bigger than mean
            (1, 5, 4, 3)  # 2nd relevant
        ]
        
        # Act
        result = get_mrr(predictions)
        
        # Assert: 1/2 = 0.5
        self.assertEqual(result, 0.5)

    def test_get_mrr_third_rank(self):
        # Arrange: Relevant item at rank 3
        predictions = [
            (1, 2, 5, 3), # actual not bigger than mean
            (1, 2, 4, 3), # actual not bigger than mean
            (1, 5, 3, 3)  # 3nd relevant
        ]
        
        # Act
        result = get_mrr(predictions)
        
        # Assert:
        self.assertAlmostEqual(result, 0.3333333333)

    def test_get_mrr_no_relevant_items(self):
        # Arrange:
        predictions = [
            (1, 2, 4, 3), # actual not bigger than mean
            (1, 4, 2, 3), # predicted not bigger than mean
        ]
        
        # Act
        result = get_mrr(predictions)
        
        # Assert:
        self.assertEqual(result, 0)

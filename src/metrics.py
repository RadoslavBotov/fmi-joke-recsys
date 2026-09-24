import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.utils import get_predictions_per_user

# Define a type alias for readability
# (user_id, actual_rating, predicted_rating, user_mean)
PredictionTuple = tuple[int, float, float, float]

def get_rmse(predictions: list[PredictionTuple]) -> float:
    """
    Calculates the Root Mean Squared Error (RMSE) for the given predictions.
    """
    actual = [p[1] for p in predictions]
    pred = [p[2] for p in predictions]
    return np.sqrt(mean_squared_error(actual, pred))

def get_mae(predictions: list[PredictionTuple]) -> float:
    """
    Calculates the Mean Absolute Error (MAE) for the given predictions.
    """
    actual = [p[1] for p in predictions]
    pred = [p[2] for p in predictions]
    return mean_absolute_error(actual, pred)

def get_mrr(predictions: list[PredictionTuple]) -> float:
    """
    Calculates the Mean Reciprocal Rank (MRR) for the recommendations.
    
    A joke is considered 'relevant' if both the actual rating and the 
    estimated rating exceed the user's mean rating from the training set.
    """
    # Group predictions by user to evaluate ranking per user
    predictions_per_user = get_predictions_per_user(predictions)

    mrr = 0
    for _, (user_mean, estimates) in predictions_per_user.items():
        # Sort predictions by estimate highest to lowest
        estimates_decs = sorted(estimates, key=lambda x: x[1], reverse=True)

        # Find the rank of the first relevant item
        for i, (act, est) in enumerate(estimates_decs):
            if (act + 1e-6 > user_mean) and (est + 1e-6 > user_mean):
                # Relevant item found at rank (i + 1)
                mrr += 1 / (i + 1)
                break

    return mrr / len(predictions_per_user)

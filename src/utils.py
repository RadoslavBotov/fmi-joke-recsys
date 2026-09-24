import pandas as pd


def get_mean_for_users(df: pd.DataFrame):
    """
    Calculates the average rating for each unique user in the dataset.
    """
    return df.groupby('user_id')['rating'].mean().to_dict()

def get_predictions_per_user(predictions: list[tuple[int, float, float, float]]) -> dict[int, tuple[float, list[tuple[float, float]]]]:
    """
    Organizes raw prediction tuples into a structured dictionary grouped by user_id.
    """
    predictions_per_user = {}
    for user_id, actual, estimate, user_mean in predictions:
        if user_id not in predictions_per_user:
            predictions_per_user[user_id] = (user_mean, [])
        predictions_per_user[user_id][1].append((actual, estimate))
    return predictions_per_user

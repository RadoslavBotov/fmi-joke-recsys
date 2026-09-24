import pandas as pd
from typing import Dict, List
from surprise import Dataset, Prediction, Reader, accuracy, NormalPredictor, SVD, KNNBasic, KNNWithMeans, AlgoBase
from surprise.model_selection import train_test_split
from surprise.trainset import Trainset

from src.metrics import PredictionTuple, get_mrr
from src.utils import get_mean_for_users


# Prediction(user_id, item_id, real_rating, estimate, details) to (user_id, user_mean, estimate)
def translate_surprise_predictions(predictions: list[Prediction], user_means: dict[int, float]) -> list[PredictionTuple]:
    """
    Translates raw Surprise Prediction objects into a standardized format for metric calculation.
    """
    return [
        (pred.uid, pred.r_ui, pred.est, user_means[pred.uid])
        for pred in predictions
    ]

def calculate_algo(algo: AlgoBase, train: Trainset, test: List[Prediction], user_means: Dict[int, float]) -> None:
    """
    Fits a Surprise algorithm, evaluates RMSE/MAE, and calculates MRR.
    """
    print(f'> {algo.__class__.__name__}:')
    
    # Train and predict
    algo.fit(train)
    predictions = algo.test(test)

    # Calculate standard accuracy metrics
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    # Calculate custom MRR metric
    predictions = translate_surprise_predictions(predictions, user_means)
    return rmse, mae, get_mrr(predictions)

'''
A Collaborative-Based approach with SVD for recommending jokes
'''
def svd_calcs(args_dict):
    """
    Collaborative filtering pipeline using SVD and KNN variants.
    """
    # Load data
    jester_data_svd_df = pd.read_csv(args_dict['data'])

    # Define rating scale for the Jester dataset
    reader = Reader(rating_scale=(-10, 10))

    print('Building dataset ...')
    data = Dataset.load_from_df(jester_data_svd_df, reader)

    print('Splitting train and test ...')
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    
    # Calculate user means from the training set
    user_means = get_mean_for_users(pd.DataFrame(train.all_ratings(), columns=['user_id', 'joke_id', 'rating']))

    print(f'> Metrics for \'{args_dict['data']}\'')

    # Evaluate multiple algorithms
    rmse, mae, mrr = calculate_algo(NormalPredictor(), train, test, user_means)
    print('RMSE: {0}\nMAE: {1}\nMRR: {2}'.format(rmse, mae, mrr))
    rmse, mae, mrr = calculate_algo(SVD(), train, test, user_means)
    print('RMSE: {0}\nMAE: {1}\nMRR: {2}'.format(rmse, mae, mrr))
    
    sim_options = {
        "name": "cosine",
        "user_based": False,  # compute similarities between items; user_based is too big for RAM
    }

    rmse, mae, mrr = calculate_algo(KNNBasic(sim_options=sim_options), train, test, user_means)
    print('RMSE: {0}\nMAE: {1}\nMRR: {2}'.format(rmse, mae, mrr))
    rmse, mae, mrr = calculate_algo(KNNWithMeans(sim_options=sim_options), train, test, user_means)
    print('RMSE: {0}\nMAE: {1}\nMRR: {2}'.format(rmse, mae, mrr))

# > Metrics for 'data_svd\jester_data_1_svd.csv':
# > NormalPredictor
# RMSE: 7.2568
# MAE:  5.8887
# MRR:  0.6516
# > SVD
# RMSE: 4.3038
# MAE:  3.3106
# MRR:  0.7395
# > KNNBasic
# RMSE: 4.2783
# MAE:  3.3546
# MRR:  0.5995
# > KNNWithMeans
# RMSE: 4.1949
# MAE:  3.2765
# MRR:  0.6546

# > Metrics for 'data_svd\jester_data_2_svd.csv':
# > NormalPredictor
# RMSE: 7.4441
# MAE:  6.0686
# MRR:  0.5547
# > SVD
# RMSE: 4.4217
# MAE:  3.5148
# MRR:  0.6027
# > KNNBasic
# RMSE: 4.4932
# MAE:  3.5445
# MRR:  0.5473
# > KNNWithMeans
# RMSE: 4.4215
# MAE:  3.4543
# MRR:  0.5850

# > Metrics for 'data_svd\jester_data_comb_svd.csv':
# > NormalPredictor
# RMSE: 7.2863
# MAE:  5.9192
# MRR:  0.6198
# > SVD
# RMSE: 4.4789
# MAE:  3.4419
# MRR:  0.6886
# > KNNBasic
# RMSE: 4.3123
# MAE:  3.3834
# MRR:  0.5824
# > KNNWithMeans
# RMSE: 4.2313
# MAE:  3.3050
# MRR:  0.6273

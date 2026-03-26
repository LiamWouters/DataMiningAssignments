## PROVIDED DATA ##
# ratings_train.csv: userId,movieId,rating,timestamp
# movies.csv: movieId,title,genres
###################

## EVENTUAL PREDICTION ##
# ratings_test.csv: userId,recommendation1,recommendation2,recommendation3,recommendation4,recommendation5,recommendation6,recommendation7,recommendation8,recommendation9,recommendation10
#########################

# TASK 1: Create a recommendation model using a COLLABORATIVE FILTERING approach
# - User/Item based nearest neighbour
# - Matrix factorization

import pandas as pd
import matplotlib.pyplot as plt
import time
from collections import defaultdict
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split, GridSearchCV
from surprise.accuracy import rmse, mae
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.prediction_algorithms.matrix_factorization import SVD

MOVIES_PATH = "../data/movies.csv"
RATINGS_TRAIN_PATH = "../data/ratings_train.csv"
PROCESSED_DATA_PATH = "../processed_data/"

def generateDataset(data, rating_scale=(1,5)):
    reader = Reader(rating_scale=rating_scale)
    return Dataset.load_from_df(data, reader=reader)

def create_KNNWithMeans(k=40, min_k=1, user_based=True):
    """
        Create KNNWithMeans model
        
        param:
            - k: int, default=40
            - min_k: int, default=1
            - user_based: boolean, default=True (if False, item based is used)
    """
    sim_options = { # Specify the similarity measure options 
        # https://surprise.readthedocs.io/en/stable/prediction_algorithms.html#similarity-measures-configuration
        "name": "cosine",               # use cosine similarity
        "user_based": user_based,       # user based (K) nearest neighbours or item based (if False)
    }
    
    return KNNWithMeans(k=k, min_k=min_k, sim_options=sim_options)

def create_model(data, type="KNN", model_args={}, rating_scale=(1,5), train_test_percent=0.2, verbose=True):
    """
        Create model and train (fit) on data.
        
        param:
            - data: data to train on (must be three columns in the order of: ["userID", "itemID", "rating"]) (https://surprise.readthedocs.io/en/stable/getting_started.html)
            - type: string, either 'KNN' for K Nearest Neighbours or 'MF' for Matrix Factorization
            - model_args: dict of arguments for each model type
            - rating_scale: tuple (range) to define the min and max rating, default=(1,5)
            - train_test_percent: float, define how much percentage of the data is split into the test set, default=0.2
            - verbose: boolean, enable/disable prints
            
        return:
            - tuple: (model (KNNWithMeans), rmse, mae)
    """
    model = None
    if type=="KNN":
        model = create_KNNWithMeans(**model_args)
    elif type=="MF":
        model = SVD(**model_args)
    else:
        print("ERROR: type can only be 'KNN' or 'MF'")
        return

    # Custom dataset requires a reader to be defined
    if verbose: print("\t-> loading data...")
    modelData = generateDataset(data, rating_scale=rating_scale)
    
    trainSet, testSet = train_test_split(modelData, train_test_percent)
    
    if verbose: print("\t-> fitting model...")
    model.fit(trainSet)
    
    ## Evaluation
    if verbose: print("\t-> evaluating model on test set...")
    predictions = model.test(testSet)
    
    # RMSE, MAE
    rootMeanSquaredError = rmse(predictions)
    meanAbsoluteError = mae(predictions)

    # Precision@K, Recall@K (K=10)
    precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=3.5)
    overallPrecision = sum(precisions.values()) / len(precisions)
    overallRecall = sum(recalls.values()) / len(recalls)
    
    if verbose:
        print(f"\t   * Root Mean Squared Error: {rootMeanSquaredError}")
        print(f"\t   * Mean Absolute Error: {meanAbsoluteError}")
        print(f"\t   * Precision@K (10): {overallPrecision}")
        print(f"\t   * Recall@K (10): {overallRecall}")
    
    return (model, rootMeanSquaredError, meanAbsoluteError, overallPrecision, overallRecall)

def try_different_K_for_model(k_values: list[int], count, data):
    results = {}
    for k in k_values:
        k_rmse_results = []
        k_mae_results = []
        k_prec_results = []
        k_rec_results = []
        
        # Run each k value 'count' times to get the mean afterward
        for i in range(count):
            model_k, model_k_rmse, model_k_mae, model_k_precision, model_k_recall = create_model(
                data=data,
                type="KNN",
                model_args={"k": k},
                verbose=False
            )
            k_rmse_results.append(model_k_rmse)
            k_mae_results.append(model_k_mae)
            k_prec_results.append(model_k_precision)
            k_rec_results.append(model_k_recall)
        
        results[k] = {"rmse": sum(k_rmse_results)/count, "mae": sum(k_mae_results)/count, "prec": sum(k_prec_results)/count, "rec": sum(k_rec_results)/count}
    
    # Plot results
    rmse_vals = [result["rmse"] for result in results.values()]
    mae_vals = [result["mae"] for result in results.values()]
    prec_vals = [result["prec"] for result in results.values()]
    rec_vals = [result["rec"] for result in results.values()]
    
    # Plot 1: RMSE, MAE
    plt.figure()
    plt.plot(k_values, rmse_vals, marker='o', linestyle='-', color='b', label='RMSE')
    plt.plot(k_values, mae_vals, marker='s', linestyle='--', color='r', label='MAE')
    plt.title('KNN Model Error for different K (lower is better)')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Error Score')
    plt.xticks(k_values)
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.legend()
    plt.grid(True)
    plt.savefig(PROCESSED_DATA_PATH + "k_evaluation_results_UserBasedKNN_rmse_mae.png")
    
    # Plot 2: precision@K, recall@K
    plt.figure()
    plt.plot(k_values, prec_vals, marker='o', linestyle='-', color='b', label='Precision@K (10)')
    plt.plot(k_values, rec_vals, marker='s', linestyle='--', color='r', label='Recall@K (10)')
    plt.title('KNN Model Precision & Recall for different K (higher is better)')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Precision & Recall score')
    plt.xticks(k_values)
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.legend()
    plt.grid(True)
    plt.savefig(PROCESSED_DATA_PATH + "k_evaluation_results_UserBasedKNN_prec_req.png")
    
    # Export to file
    results_df = pd.DataFrame.from_dict(results,orient="index")
    results_df.index.name = "K_value"
    results_df.to_csv(PROCESSED_DATA_PATH + "k_evaluation_results_UserBasedKNN.csv")

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """
        Return precision and recall at k metrics for each user
        
        IMPORTANT: This method is not self-implemented but taken from the surprise documantation:
                    https://surprise.readthedocs.io/en/stable/FAQ.html#how-to-compute-precision-k-and-recall-k
    """
    
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls


if __name__ == "__main__":
    # TASK 1 FLAGS:
    CREATE_EXAMPLE_MODELS = False
    EVALUATE_KNN = False
    EVALUATE_MF = True
    # TASK 2 FLAGS:
    
    ##########################################
    # Files holding data
    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_TRAIN_PATH)
    
    # PRE-PROCESS (surprise only takes [userId, itemId, rating] anyways)
    print("Pre processing data...")
    print("  -> Merging tables")
    combined = pd.merge(ratings, movies, on="movieId", how="left")
    print("  -> One-Hot encode categorical variables (genres)") 
    # https://www.geeksforgeeks.org/machine-learning/ml-one-hot-encoding/
    # https://www.geeksforgeeks.org/python/python-pandas-series-str-get_dummies/
    one_hot_encoded_genres = combined["genres"].str.get_dummies(sep='|')
    combined = pd.concat([combined, one_hot_encoded_genres], axis=1)
    combined = combined.drop("genres", axis=1)
    
    combined.to_csv(PROCESSED_DATA_PATH + "combined.csv")
    print("Pre processing done!")
    
    if CREATE_EXAMPLE_MODELS:
        print("Creating first model... (Collaborative Filtering, user based nearest neighbour)")
        model1, model1rmse, model1mae = create_model(
            data=ratings[["userId", "movieId", "rating"]],
            type="KNN"
        )
        print("Model created!")
    
    if EVALUATE_KNN: # What K is best? Is there a significant difference?
        ## The following function will try a range of K_values, each 'count' amount of times and get the mean rmse and mae
        try_different_K_for_model(k_values=[5,20,35,50,65,80,95,110,125,140], count=10, data=ratings[["userId", "movieId", "rating"]])
    
    if CREATE_EXAMPLE_MODELS:
        print("Creating second model... (Collaborative Filtering, item based nearest neighbour)")
        model2, model2rmse, model2mae = create_model(
            data=ratings[["userId", "movieId", "rating"]],
            type="MF",
            model_args={
                "n_factors":100,
                "n_epochs":20
            }
        )
        print("Model created!")
    
    if EVALUATE_MF: # Determine influence of parameters (and parameter combinations)
        param_grid = {
            "n_factors": [200, 300, 400, 500, 600],
            "n_epochs": [50, 70, 90, 110, 130],
            "lr_all": [0.01],#[0.005, 0.01, 0.015],
            "reg_all": [0.1]#[0.05, 0.1, 0.015]
        }
        # param_grid = {
        #     "n_factors": [150, 200, 250, 300, 350, 400],
        #     "n_epochs": [50, 60, 70, 80, 90],
        #     "lr_all": [0.01],#[0.005, 0.01, 0.015],
        #     "reg_all": [0.1]#[0.05, 0.1, 0.015]
        # }
        print("RUNNING GRIDSEARCHCV (will take a while)")
        timeBefore = time.time()
        gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=3)
        gs.fit(generateDataset(ratings[["userId", "movieId", "rating"]]))
        print(f"GRIDSEARCHCV took {time.time() - timeBefore} seconds")
        print(f"Best RMSE score: {gs.best_score['rmse']} (params: {gs.best_params['rmse']})")
        print(f"Best MAE score: {gs.best_score['mae']} (params: {gs.best_params['mae']})")
        
        print("Evaluating ranking metrics for best MF model...")
        best_rmse_mf_model, mf_bestRMSE_rmse, mf_bestRMSE_mae, mf_bestRMSE_prec, mf_bestRMSE_recall = create_model(
            data=ratings[["userId", "movieId", "rating"]],
            type="MF",
            model_args=gs.best_params['rmse']
        )
        best_mae_mf_model, mf_bestMAE_rmse, mf_bestMAE_mae, mf_bestMAE_prec, mf_bestMAE_recall = create_model(
            data=ratings[["userId", "movieId", "rating"]],
            type="MF",
            model_args=gs.best_params['mae']
        )
        print("Best RMSE model:\n  -> Precision@K (K=10):", mf_bestRMSE_prec, "\n  -> Recall@K (K=10):", mf_bestRMSE_recall)
        print("Best MAE model:\n  -> Precision@K (K=10):", mf_bestMAE_prec, "\n  -> Recall@K (K=10):", mf_bestMAE_recall)

    # TODO: (Task 2) generate top 10 for each user with the best performing model.
    

## PROVIDED DATA ##
# ratings_train.csv: userId,movieId,rating,timestamp
# movies.csv: movieId,title,genres
###################

# TASK 1: Create a recommendation model using a COLLABORATIVE FILTERING approach
# - User/Item based nearest neighbour
# - Matrix factorization

import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse, mae
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.prediction_algorithms.matrix_factorization import SVD

MOVIES_PATH = "../data/movies.csv"
RATINGS_TRAIN_PATH = "../data/ratings_train.csv"
PROCESSED_DATA_PATH = "../processed_data/"

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
    reader = Reader(rating_scale=rating_scale)
    modelData = Dataset.load_from_df(data, reader=reader)
    
    trainSet, testSet = train_test_split(modelData, train_test_percent)
    
    if verbose: print("\t-> fitting model...")
    model.fit(trainSet)
    
    if verbose: print("\t-> evaluating model on test set...")
    predictions = model.test(testSet)
    rootMeanSquaredError = rmse(predictions)
    if verbose: print(f"\t   * Root Mean Squared Error: {rootMeanSquaredError}")
    meanAbsoluteError = mae(predictions)
    if verbose: print(f"\t   * Mean Absolute Error: {meanAbsoluteError}")
       
    return (model, rootMeanSquaredError, meanAbsoluteError)

def try_different_K_for_model(k_values: list[int], count, data):
    results = {}
    for k in k_values:
        k_rmse_results = []
        k_mae_results = []
        
        # Run each k value 'count' times to get the mean afterward
        for i in range(count):
            model_k, model_k_rmse, model_k_mae = create_model(
                data=data,
                type="KNN",
                model_args={"k": k},
                verbose=False
            )
            k_rmse_results.append(model_k_rmse)
            k_mae_results.append(model_k_mae)
        
        results[k] = {"rmse": sum(k_rmse_results)/count, "mae": sum(k_mae_results)/count}
    
    # Plot results
    rmse_vals = [result["rmse"] for result in results.values()]
    mae_vals = [result["mae"] for result in results.values()]
    plt.figure()
    plt.plot(k_values, rmse_vals, marker='o', linestyle='-', color='b', label='RMSE')
    plt.plot(k_values, mae_vals, marker='s', linestyle='--', color='r', label='MAE')
    plt.title('User based KNN Model Error for different K')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Error Score')
    plt.xticks(k_values)
    plt.legend()
    plt.grid(True)
    plt.savefig(PROCESSED_DATA_PATH + "k_evaluation_results_UserBasedKNN.png")
    
    # Export to file
    results_df = pd.DataFrame.from_dict(results,orient="index")
    results_df.index.name = "K_value"
    results_df.to_csv(PROCESSED_DATA_PATH + "k_evaluation_results_UserBasedKNN.csv")
    

if __name__ == "__main__":
    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_TRAIN_PATH)
    
    print("Creating first model... (Collaborative Filtering, user based nearest neighbour)")
    model1, model1rmse, model1mae = create_model(
        data=ratings[["userId", "movieId", "rating"]],
        type="KNN"
    )
    print("Model created!")
    
    ## What K is best? Is there a significant difference?
    ## The following function will try a range of K_values, each 'count' amount of times and get the mean rmse and mae
    ## UNCOMMENT THE NEXT LINE TO RE-RUN:
    # try_different_K_for_model(k_values=[5,20,35,50,65,80,95,110,125,140], count=10, data=ratings[["userId", "movieId", "rating"]])
    
    
        
    print("Creating second model... (Collaborative Filtering, item based nearest neighbour)")
    model2, model2rmse, model2mae = create_model(
        data=ratings[["userId", "movieId", "rating"]],
        type="MF",
        model_args={
            "n_factors":100,
            "n_epochs":20,
            "random_state":42
        }
    )
    print("Model created!")
    
    # TODO: finetune parameters of the MF model
    # TODO: find best model, and continue with task 2

# PRE-PROCESS
#combined = pd.merge(ratings, movies, on="movieId", how="left")
#combined.to_csv(PROCESSED_DATA_PATH + "combined.csv")µ

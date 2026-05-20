## PROVIDED DATA ##
# income.csv:       age,workclass,education,marital status,occupation,workinghours,sex,ability to speak english,gave birth this year,income
# income_test.csv:  age,workclass,education,marital status,occupation,workinghours,sex,ability to speak english,gave birth this year
###################

## EVENTUAL PREDICTION ##
# predictions_template.csv: id,income
#########################

import time, shap, pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src import Preprocessor, DecisionTree, RandomForest, KNearestNeighbors, GaussianNaiveBayes

filePath = pathlib.Path(__file__).parent.resolve()

# DATA PATH PARAMETERS
## PLACE THE ASSIGNMENT (INPUT) DATA FILES HERE
INCOME_TEST_PATH = "./data/income_test.csv"
INCOME_PATH = "./data/income.csv"
PREDICTIONS_TEMPLATE_PATH = "./data/predictions_templace.csv"
## OUTPUT FILES WILL BE GENERATED HERE
PROCESSED_DATA_PATH = "./processed_data/"

if __name__ == "__main__": 
    # PREPROCESSING FLAGS   
    DROP_ABILITY_TO_SPEAK_ENGLISH = True    # if not dropped its empty rows will be filled and the column will be normalized
    DROP_GAVE_BIRTH_THIS_YEAR = True        # ^^ and the column will be one hot encoded instead of normalized
    FILL_CONDITION_GAVE_BIRTH_THIS_YEAR = True # Ignored if the column is dropped with DROP_GAVE_BIRTH_THIS_YEAR

    NORMALIZE_COLUMNS = ["age", "education", "workinghours"]  # Columns of which the values must be normalized from: [min_value, max_value] -> [0, 1]
    OHE_COLUMNS = ["workclass", "marital status", "occupation", "sex"] # Define columns to one hot encode
    LABEL_COLUMN = "income"
    
    # CHOSEN MODEL (Random Forest)
    CHOSEN_MODEL_PARAMETERS = { # Parameters for the model chosen for Task 2 & 3 (Random Forest)
        "max_depth": 20, "min_samples_leaf": 5, "min_samples_split": 20, "n_estimators": 250
    }
    
    # GENERAL EXECUTION FLAGS
    PERFORM_INITIAL_MODEL_CREATION = True
    PERFORM_GRIDSEARCHCV = True
    PERFORM_BEST_GRIDSEARCH_MODEL_CREATION = True
    PERFORM_SHAP_COMPUTE = True
    PERFORM_FINAL_CLASSIFICATION = True
    
    RANDOM_SEED = 1 #int(time.time())
    #################
    
    pp = Preprocessor()
    
    # Handle missing data: "ability to speak english", "gave birth this year"
    ## 0. print counts of how many have entries
    # tempDf = pd.read_csv(INCOME_PATH)
    # print(f"Income training data has {tempDf['ability to speak english'].isna().sum()} empty rows in column 'ability to speak english' out of {len(tempDf['ability to speak english'])} total rows")
    # print(f"Income training data has {tempDf['gave birth this year'].isna().sum()} empty rows in column 'gave birth this year' out of {len(tempDf['gave birth this year'])} total rows")
    ### Column: "ability to speak english"
    if DROP_ABILITY_TO_SPEAK_ENGLISH:
        ## 1. drop the column as 8597/9000 rows are empty in "income.csv"
        pp.DropColumns(["ability to speak english"])
    else:
        ## 2. fill all empty with 0 (all existing values in the column currently are either [NaN, 1.0, 2.0, 3.0, 4.0])
        pp.FillEmpty("ability to speak english", 0.0)
        
        # Normalize the numerical column
        NORMALIZE_COLUMNS.append("ability to speak english")
    ### Column: "gave birth this year"
    if DROP_GAVE_BIRTH_THIS_YEAR:
        ## 1. drop the column as 7058/9000 rows are empty in "income.csv"
        pp.DropColumns(["gave birth this year"])
    else:
        if not FILL_CONDITION_GAVE_BIRTH_THIS_YEAR:
            ## 2. fill all empty with no
            pp.FillEmpty("gave birth this year", value="No")
        else:
            ## 3. fill all male empty with no, fill female
            def fill_condition(row):
                if pd.isna(row["gave birth this year"]):
                    if row["sex"] == "Male":
                        return "No"
                    else:
                        return "Maybe"
                return row["gave birth this year"]
            pp.FillEmpty("gave birth this year", condition=fill_condition)
        
        # make sure to also encode as categorized
        OHE_COLUMNS.append("gave birth this year")
    
    # Normalize columns
    pp.Normalize(columns=NORMALIZE_COLUMNS)
    
    # Encode categorical values
    pp.OneHotEncode(columns=OHE_COLUMNS)
    
    # Encode labels
    pp.LabelEncode(column=LABEL_COLUMN)
    
    # Perform all the set preprocessings
    pp.process(INCOME_PATH, INCOME_TEST_PATH, save_to_file=True)
    
    if PERFORM_INITIAL_MODEL_CREATION:
        dt = DecisionTree(pp, "income", random_seed=RANDOM_SEED)
        print(f"[DecisionTree] Train Accuracy: {dt.train_accuracy}")
        print(f"[DecisionTree] Test Accuracy: {dt.test_accuracy}")
        print(f"[DecisionTree] (roc) AUC: {dt.roc_auc}")
        print(f"[DecisionTree] Precision: {dt.precision}")
        print(f"[DecisionTree] Recall: {dt.recall}")
        rf = RandomForest(pp, "income", random_seed=RANDOM_SEED)
        print(f"[RandomForest] Train Accuracy: {rf.train_accuracy}")
        print(f"[RandomForest] Test Accuracy: {rf.test_accuracy}")
        print(f"[RandomForest] (roc) AUC: {rf.roc_auc}")
        print(f"[RandomForest] Precision: {rf.precision}")
        print(f"[RandomForest] Recall: {rf.recall}")
        knn = KNearestNeighbors(pp, "income", random_seed=RANDOM_SEED)
        print(f"[KNearestNeighbors] Train Accuracy: {knn.train_accuracy}")
        print(f"[KNearestNeighbors] Test Accuracy: {knn.test_accuracy}")
        print(f"[KNearestNeighbors] (roc) AUC: {knn.roc_auc}")
        print(f"[KNearestNeighbors] Precision: {knn.precision}")
        print(f"[KNearestNeighbors] Recall: {knn.recall}")
        gnb = GaussianNaiveBayes(pp, "income", random_seed=RANDOM_SEED)
        print(f"[GaussianNaiveBayes] Train Accuracy: {gnb.train_accuracy}")
        print(f"[GaussianNaiveBayes] Test Accuracy: {gnb.test_accuracy}")
        print(f"[GaussianNaiveBayes] (roc) AUC: {gnb.roc_auc}")
        print(f"[GaussianNaiveBayes] Precision: {gnb.precision}")
        print(f"[GaussianNaiveBayes] Recall: {gnb.recall}")
    
    if PERFORM_GRIDSEARCHCV:
        print("\nPerform Grid Searches:")
        # Grid search for Decision Tree
        dt_param_grid = {
            "max_depth": [None, 4, 6, 8, 10, 15, 20],
            "max_features": [None, "sqrt", "log2"],
            "min_samples_split": [5, 10, 15, 20],  
            "min_samples_leaf": [5, 10, 15, 20],
        }
        dt_gscv = DecisionTree(pp, "income", random_seed=RANDOM_SEED)
        
        timeBefore = time.time()
        best_dt_params = dt_gscv.cross_validate(dt_param_grid, score_metric="accuracy")
        timeAfter = time.time()
        print(f"[[Decision Tree Gridsearch took: {timeAfter-timeBefore} seconds]]")
            
        # Grid search for Random Forest
        rf_param_grid = {
            "n_estimators": [100, 150, 200, 250],
            "max_depth": [5, 10, 25, 20],
            "min_samples_split": [10, 20],  
            "min_samples_leaf": [5, 10],
        }
        rf_gscv = RandomForest(pp, "income", random_seed=RANDOM_SEED)
        
        timeBefore = time.time()
        best_rf_params = rf_gscv.cross_validate(rf_param_grid, score_metric="accuracy")
        timeAfter = time.time()
        print(f"[[Random Forest Gridsearch took: {timeAfter-timeBefore} seconds]]")
        
        if PERFORM_BEST_GRIDSEARCH_MODEL_CREATION:
            best_dt = DecisionTree(pp, "income", model_params=best_dt_params, random_seed=RANDOM_SEED)
            print(f"Best params: [DecisionTree] Train Accuracy: {best_dt.train_accuracy}")
            print(f"Best params: [DecisionTree] Test Accuracy: {best_dt.test_accuracy}")
            print(f"Best params: [DecisionTree] (roc) AUC: {best_dt.roc_auc}")
            print(f"Best params: [DecisionTree] Precision: {best_dt.precision}")
            print(f"Best params: [DecisionTree] Recall: {best_dt.recall}")
            
            best_rf = RandomForest(pp, "income", model_params=best_rf_params, random_seed=RANDOM_SEED)
            print(f"Best params: [RandomForest] Train Accuracy: {best_rf.train_accuracy}")
            print(f"Best params: [RandomForest] Test Accuracy: {best_rf.test_accuracy}")
            print(f"Best params: [RandomForest] (roc) AUC: {best_rf.roc_auc}")
            print(f"Best params: [RandomForest] Precision: {best_rf.precision}")
            print(f"Best params: [RandomForest] Recall: {best_rf.recall}")

    chosen_model = RandomForest(pp, "income", model_params=CHOSEN_MODEL_PARAMETERS, random_seed=RANDOM_SEED)
    chosen_model._train_model()
    # print(f"Feature Importances: {chosen_model.importances}")
    
    # SHAP
    if PERFORM_SHAP_COMPUTE:
        print("Calculating SHAP values")
        timeBefore = time.time()
        explainer = shap.TreeExplainer(chosen_model.model)
        shap_values = explainer(chosen_model._X_test)
        timeAfter = time.time()
        print(f"[[SHAP COMPUTE TIME: {timeAfter-timeBefore} seconds]")

        shap_highincome = shap_values[:, :, 0]
        plt.clf()
        shap.plots.beeswarm(
            shap_highincome,
            show=False
        )
        plt.tight_layout()
        plt.savefig((filePath / "../processed_data/shap_beeswarm").resolve())
        data = pd.read_csv(INCOME_PATH)
        for waterfall_index in range(10):
            person_test_data = chosen_model._X_test.iloc[waterfall_index]
            person_test_data_df = chosen_model._X_test.iloc[[waterfall_index]]
            person_index = person_test_data.name
            person_full_row = data.iloc[person_index]
            prediction = pp._label_encoder.inverse_transform(chosen_model.predict_single(person_test_data_df))[0]
        
            chosen_model.predict()

            plt.clf()
            shap.plots.waterfall(shap_highincome[waterfall_index], show=False)
            plt.suptitle(f"[SHAP] person: {person_index}\nreal income: {person_full_row['income']}, predicted income: {prediction}")
            plt.tight_layout()
            plt.savefig((filePath / f"../processed_data/shap_waterfall_{waterfall_index}").resolve())

    if PERFORM_FINAL_CLASSIFICATION:
        chosen_model.predict()

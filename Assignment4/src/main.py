## PROVIDED DATA ##
# articles.csv: doc_id,text
###################

## EVENTUAL PREDICTION ##
# anomalies.csv: anomaly,doc_id
# clusters.csv: doc_id,label
#########################

import pandas as pd
from . import Preprocessor, Classifier

# DATA PATH PARAMETERS
## PLACE THE ASSIGNMENT (INPUT) DATA FILES HERE
INCOME_TEST_PATH = "./data/income_test.csv"
INCOME_PATH = "./data/income.csv"
PREDICTIONS_TEMPLATE_PATH = "./data/predictions_templace.csv"
## OUTPUT FILES WILL BE GENERATED HERE
PROCESSED_DATA_PATH = "./processed_data/"

if __name__ == "__main__":    
    DROP_ABILITY_TO_SPEAK_ENGLISH = False    # if not dropped its empty rows will be filled and the column will be normalized
    DROP_GAVE_BIRTH_THIS_YEAR = False        # ^^ and the column will be one hot encoded instead of normalized
    FILL_CONDITION_GAVE_BIRTH_THIS_YEAR = True # Ignored if the column is dropped with DROP_GAVE_BIRTH_THIS_YEAR

    NORMALIZE_COLUMNS = ["age", "education", "workinghours"]  # Columns of which the values must be normalized from: [min_value, max_value] -> [0, 1]
    
    OHE_COLUMNS = ["workclass", "marital status", "occupation", "sex"] # Define columns to one hot encode
    
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
    
    # Perform all the set preprocessings
    pp.process(INCOME_PATH, INCOME_TEST_PATH, ignore_from_train_data="income", save_to_file=True)
    
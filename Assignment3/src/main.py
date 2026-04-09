## PROVIDED DATA ##
# articles.csv: doc_id,text
###################

## EVENTUAL PREDICTION ##
# anomalies.csv: anomaly,doc_id
# clusters.csv: doc_id,label
#########################

import pandas as pd
import matplotlib.pyplot as plt

# DATA PATH PARAMETERS
## PLACE THE ASSIGNMENT (INPUT) DATA FILES HERE
ANOMALIES_PATH = "../data/anomalies.csv"
ARTICLES_PATH = "../data/articles.csv"
CLUSTERS_PATH = "../data/clusters.csv"
## OUTPUT FILES WILL BE GENERATED HERE
PROCESSED_DATA_PATH = "../processed_data/"

if __name__ == "__main__":
    pass

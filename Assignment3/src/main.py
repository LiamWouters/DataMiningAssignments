## PROVIDED DATA ##
# articles.csv: doc_id,text
###################

## EVENTUAL PREDICTION ##
# anomalies.csv: anomaly,doc_id
# clusters.csv: doc_id,label
#########################

import pandas as pd
import matplotlib.pyplot as plt
import os
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans

# DATA PATH PARAMETERS
## PLACE THE ASSIGNMENT (INPUT) DATA FILES HERE
ANOMALIES_PATH = "../data/anomalies.csv"
ARTICLES_PATH = "../data/articles.csv"
CLUSTERS_PATH = "../data/clusters.csv"
## OUTPUT FILES WILL BE GENERATED HERE
PROCESSED_DATA_PATH = "../processed_data/"
## NLTK DATA WILL BE DOWNLOADED HERE (if using lemmatization)
NLTK_DATA_PATH = "../nltk_data/"

####### HELPER #######
class LemmaTokenizer:
    # From: http://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
    def __init__(self, pos_tagging: bool):
        self.wnl = WordNetLemmatizer()
        self.token_regex = r"(?u)\b\w\w+\b" # Default from sklearns CountVectorizer token_pattern
        self.pos_tagging = pos_tagging
        
    def get_wordnet_pos(self,treebank_tag): 
        # source for method: https://stackoverflow.com/a/15590384
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # default pos in lemmatization is Noun
            return wordnet.NOUN
    
    def __call__(self, doc):
        # Regular tokenization using the same regex as the default from CountVectorizer
        tokenizer = RegexpTokenizer(self.token_regex)
        tokens = tokenizer.tokenize(doc)
        
        if self.pos_tagging:
            # Apply pos (part-of-speech) tagging for better lemmatization
            tokens = nltk.pos_tag(tokens)
            return [self.wnl.lemmatize(t[0],self.get_wordnet_pos(t[1])) for t in tokens]
        else: 
            # Apply lemmatization without pos_tagging
            return [self.wnl.lemmatize(t) for t in tokens]

######################

#### MAIN METHODS ####
def apply_bow(articles, use_tfidf, use_lemmatization, use_pos_tagging, use_stopwords, min_df, max_df) -> pd.DataFrame:
    """
    This function takes the raw articles data and returns a dataframe that contains the same data represented with the bag of words format
    
    The implementation is based on: https://datascientyst.com/create-a-bag-of-words-pandas-python/
    """
    stopwordlist = None
    if use_stopwords:
        stopwordlist = stopwords.words('english')
    
    vectorizer = None
    if use_tfidf:
        vectorizer = TfidfVectorizer(
            tokenizer = LemmaTokenizer(use_pos_tagging) if use_lemmatization else None,
            stop_words=stopwordlist,
            max_df=max_df,
            min_df=min_df,
        )
    else:
        vectorizer = CountVectorizer(
            tokenizer = LemmaTokenizer(use_pos_tagging) if use_lemmatization else None,
            stop_words=stopwordlist,
            max_df=max_df,
            min_df=min_df,
        )
    bow = vectorizer.fit_transform(articles['text'])
    count_array = bow.toarray()
    features = vectorizer.get_feature_names_out()
    
    return pd.DataFrame(data=count_array, columns=features, index=articles['doc_id'])

def run_clustering(data, algorithm="KMeans", n_clusters=5, model_args={}, verbose=True):
    """
    Create model, fit on data, and predict cluster labels.
    
    param:
        - data: DataFrame holding the preprocessed BOW articles
        - algorithm: string to choose which algorithm to use such as "KMeans"
        - n_clusters: int, number of clusters to form
        - model_args: dict of additional arguments for the specific model (e.g., random_state)
        - verbose: boolean, enable/disable prints
        
    return:
        - tuple: (trained_model, cluster_labels)
    """
    model = None
    
    ## Initialize the specified model
    if algorithm == "KMeans":
        if verbose: print(f"\t-> initializing {algorithm} with {n_clusters} clusters...")
        # Note: setting n_init="auto" is good practice for newer sklearn versions
        model = KMeans(n_clusters=n_clusters, **model_args)
        
    # elif algorithm == "Hierarchical":
    #     # Placeholder for future implementation
    #     pass 
        
    # elif algorithm == "Spectral":
    #     # Placeholder for future implementation
    #     pass
        
    else:
        print(f"ERROR: algorithm '{algorithm}' is not supported yet.")
        return (None, None)

    ## Fit the model and get predictions
    if verbose: print("\t-> fitting model and predicting...")
    
    labels = model.fit_predict(data)
    
    ## TODO: Evaluation
    # if verbose: print("\t-> evaluating model (TODO)...")
    
    return (model, labels)

######################


if __name__ == "__main__":
    ### FLAGS ###
    PREPROCESS_TFIDFVECTORIZER = True # If this is false, a regular CountVectorizer is used
    PREPROCESS_LEMMATIZATION = True # WARNING: this will download NLTK data
    PREPROCESS_LEMMATIZATION_POSTAGGING = True
    PREPROCESS_STOPWORDS = True
    PREPROCESS_MINDF = True
    PREPROCESS_MAXDF = True
    
    
    #############
    
    # CONSTANTS #
    MIN_DF = 0.05
    MAX_DF = 0.7
    #############
    
    articles = pd.read_csv(ARTICLES_PATH)
    
    toDownload = []
    
    if PREPROCESS_LEMMATIZATION:
        toDownload.append('wordnet')
    if PREPROCESS_LEMMATIZATION_POSTAGGING:
        toDownload.append('averaged_perceptron_tagger_eng')
    if PREPROCESS_STOPWORDS:
        toDownload.append('stopwords')
        
    if len(toDownload) > 0:
        print(f"Downloading NLTK_DATA: {toDownload}")
        if NLTK_DATA_PATH not in nltk.data.path:
            nltk.data.path.append(NLTK_DATA_PATH)
        nltk.download(toDownload, download_dir=NLTK_DATA_PATH)

    ### Preprocess ###
    print("Preprocessing articles data...")
    articles = apply_bow(
        articles,
        use_tfidf=PREPROCESS_TFIDFVECTORIZER,
        use_lemmatization=PREPROCESS_LEMMATIZATION,
        use_pos_tagging=PREPROCESS_LEMMATIZATION_POSTAGGING,
        use_stopwords=PREPROCESS_STOPWORDS,
        min_df=MIN_DF if PREPROCESS_MINDF else 1,
        max_df=MAX_DF if PREPROCESS_MAXDF else 1.0,
    )
    print("Preprocessing DONE!")
        
    # Preprocessed data
    articles.to_csv(PROCESSED_DATA_PATH + "after_preprocessing.csv")
    
    ### Clustering ###
    print("Running clustering algorithm...")
    model, labels = run_clustering(
        data=articles,
        algorithm="KMeans",
        n_clusters=5,
        model_args={
            
        },
        verbose=True
    )
    
    if labels is not None:
        print("Saving clusters...")
        clusters = pd.read_csv(CLUSTERS_PATH)
        clusters['label'] = labels
        clusters.to_csv(PROCESSED_DATA_PATH+"clusters.csv", index=False)
        print(f"Saved assigned clusters to {PROCESSED_DATA_PATH+'clusters.csv'}")
        
        print("Saving all clustered articles to their own files")
        clusterFiles_dir = os.path.join(PROCESSED_DATA_PATH, "cluster_files")
        
        if os.path.exists(clusterFiles_dir):
            # Iterate over all files in the directory and delete them (previous run)
            for filename in os.listdir(clusterFiles_dir):
                file_path = os.path.join(clusterFiles_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            os.makedirs(clusterFiles_dir)
        
        articles['cluster_label'] = labels
        unique_labels = articles['cluster_label'].unique()
        
        for label in unique_labels:
            cluster_df = articles[articles['cluster_label'] == label].copy()
            cluster_df = cluster_df.drop(columns=['cluster_label'])
            # cluster_df = cluster_df.loc[:, (cluster_df != 0).any(axis=0)] # drop collumns where all row entries are 0 (in the bag of words)
            clusterFile_path = os.path.join(clusterFiles_dir, f"cluster_{int(label)}.csv")
            cluster_df.to_csv(clusterFile_path, index_label='DOC_ID')            

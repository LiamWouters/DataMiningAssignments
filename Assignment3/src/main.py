## PROVIDED DATA ##
# articles.csv: doc_id,text
###################

## EVENTUAL PREDICTION ##
# anomalies.csv: anomaly,doc_id
# clusters.csv: doc_id,label
#########################

import pandas as pd
import matplotlib.pyplot as plt
import os, shutil, nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score

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
        - dict with keys "model, labels, silhouette(, SSE)"
    """
    model = None
    
    ## Initialize the specified model
    if verbose: print(f"\t-> initializing {algorithm} with {n_clusters} clusters...")
    if algorithm == "KMeans":
        model = KMeans(n_clusters=n_clusters, **model_args)
        
    elif algorithm == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters)
        
    elif algorithm == "Spectral":
        model = SpectralClustering(n_clusters=n_clusters)
        
    else:
        print(f"ERROR: algorithm '{algorithm}' is not supported yet.")
        return {}

    ## Fit the model and get predictions
    if verbose: print("\t-> fitting model and predicting...")
    
    labels = model.fit_predict(data)
    
    ## Evaluation
    if verbose: print("\t-> evaluating model (silhouette score)...")
    silhouette = silhouette_score(data, labels, metric='euclidean')
    
    ## Results
    results = {"model": model, "labels": labels, "silhouette": silhouette}
    if algorithm == "KMeans": 
        results["SSE"] = model.inertia_
    
    return results

######################


if __name__ == "__main__":
    ### FLAGS ###
    RANDOM_SEED = 1
    SKIP_PREPROCESS = True # If true, should have "articles_preprocessed.csv" in processed_data
    PREPROCESS_TFIDFVECTORIZER = False # If this is false, a regular CountVectorizer is used
    PREPROCESS_LEMMATIZATION = True # WARNING: this will download NLTK data
    PREPROCESS_LEMMATIZATION_POSTAGGING = True
    PREPROCESS_STOPWORDS = True
    PREPROCESS_MINDF = True
    PREPROCESS_MAXDF = True
    
    SSE_CURVE = True    # Plot the SSE graph for all cluster counts (KMeans only)
    RUN_ALGORITHMS = {  # Specify which algorithms to run (with which arguments) and with which cluster count
        "KMeans": {"modelArgs": {"n_init": 10, "random_state": RANDOM_SEED}, "clusterCounts": [i for i in range(2,11)]}, 
        "Hierarchical": {"modelArgs": {}, "clusterCounts": [i for i in range(2,11)]}, 
        # "Spectral": {"modelArgs": {"random_state": RANDOM_SEED}, "clusterCounts": [i for i in range(2,11)]}
    }
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
        
    if len(toDownload) > 0 and not SKIP_PREPROCESS:
        print(f"Downloading NLTK_DATA: {toDownload}")
        if NLTK_DATA_PATH not in nltk.data.path:
            nltk.data.path.append(NLTK_DATA_PATH)
        nltk.download(toDownload, download_dir=NLTK_DATA_PATH)

    ### Preprocess ###
    if not SKIP_PREPROCESS: 
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
        articles.to_csv(PROCESSED_DATA_PATH + "articles_preprocessed.csv")
    else:
        # Assume preprocessing is already done and do not re-do it, just load
        articles = pd.read_csv(PROCESSED_DATA_PATH + "articles_preprocessed.csv",index_col="doc_id")
        
    ### Clustering ###
    for algo in RUN_ALGORITHMS:
        algo_args = RUN_ALGORITHMS[algo]["modelArgs"]
        
        ## Make sure directory for the algorithm exists and clear it
        algo_path = os.path.join(PROCESSED_DATA_PATH, algo)
        if os.path.exists(algo_path):
            shutil.rmtree(algo_path)    # Clear all files from the previous run
        else:
            os.makedirs(algo_path)
            
        sse_scores = []
        
        for cluster_count in RUN_ALGORITHMS[algo]["clusterCounts"]:
            print(f"Running clustering algorithm {algo} for {cluster_count} clusters...")
            resultsDict = run_clustering(
                data=articles,
                algorithm=algo,
                n_clusters=cluster_count,
                model_args=algo_args,
                verbose=True
            )
            labels = resultsDict["labels"]
            
            if algo == "KMeans":
                sse_scores.append(resultsDict["SSE"])
            
            # Save output clusters
            algo_clusterCount_path = os.path.join(algo_path, f"{cluster_count}_clusters")
            os.makedirs(algo_clusterCount_path)
            
            if labels is not None:
                full_filename = f"clusters_{cluster_count}_{algo}.csv"
                full_filepath = os.path.join(algo_clusterCount_path, full_filename)
                
                print("\t=>Saving clusters...")
                clusters = pd.read_csv(CLUSTERS_PATH)
                clusters['label'] = labels
                clusters.to_csv(full_filepath, index=False)
                print(f"\t\t-> Saved assigned clusters to {full_filename}")
                
                print("\t\t-> Saving all clusters to their own files")
                articles['cluster_label'] = labels
                unique_labels = articles['cluster_label'].unique()
                
                for label in unique_labels:
                    cluster_df = articles[articles['cluster_label'] == label].copy()
                    cluster_df = cluster_df.drop(columns=['cluster_label'])
                    
                    ####################
                    # Sort the top terms for the cluster
                    specific_cluster_terms_filepath = os.path.join(algo_clusterCount_path, f"cluster_{int(label)}_terms.csv")
                    term_frequencies: pd.DataFrame = cluster_df.sum(axis=0)
                    sorted_term_frequencies = term_frequencies.sort_values(ascending=False).to_frame().transpose()
                    sorted_term_frequencies.to_csv(specific_cluster_terms_filepath,index=False)
                    ####################
                    
                    clusterFile_path = os.path.join(algo_clusterCount_path, f"cluster_{int(label)}.csv")
                    cluster_df.to_csv(clusterFile_path, index_label='DOC_ID')   
                
                # Make sure to remove the added collumn again
                articles = articles.drop(columns=['cluster_label'])
        
        if algo == "KMeans" and SSE_CURVE:
            plot_path = os.path.join(algo_path, "sse_graph.png")
            plt.figure()
            plt.plot(RUN_ALGORITHMS[algo]["clusterCounts"], sse_scores, marker='o', linestyle='-', color='b', label='SSE')
            plt.title('SSE for cluster count for KMeans (lower is better)')
            plt.xlabel('Number of clusters')
            plt.ylabel('Error Score')
            plt.xticks(RUN_ALGORITHMS[algo]["clusterCounts"])
            plt.grid(True)
            plt.savefig(plot_path)
            
    # TODO: 
    # 1. what to do with silhouette score
    # 2. compare and find the categories
    #    -> find "best" clusterer
    # 0. try spectral if it does not take too long
        

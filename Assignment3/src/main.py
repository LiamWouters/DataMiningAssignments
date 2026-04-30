## PROVIDED DATA ##
# articles.csv: doc_id,text
###################

## EVENTUAL PREDICTION ##
# anomalies.csv: anomaly,doc_id
# clusters.csv: doc_id,label
#########################

import pandas as pd
import matplotlib.pyplot as plt
import os, shutil, nltk, contractions
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
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
    def __init__(self, pos_tagging: bool, fix_contractions: bool, verbose: bool = False, 
                 token_regex = r"(?u)\b\w\w+\b" # Default from sklearns CountVectorizer token_pattern
                 ):
        self.wnl = WordNetLemmatizer()
        self.token_regex = token_regex
        self.pos_tagging = pos_tagging
        self.fix_contractions = fix_contractions
        self.verbose = verbose
        
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
        p = False
        if len(doc) < 300 and self.verbose:
            p = True
            print(f"-----------------\n\t* [FULL] {doc}")
        
        # Expand contractions such as "we've" to "we have"
        if self.fix_contractions:
            doc = contractions.fix(doc)
            if p: print(f"\t* [FIX CONTR] {doc}")
        
        # Regular tokenization using the same regex as the default from CountVectorizer
        tokenizer = RegexpTokenizer(self.token_regex)
        tokens = tokenizer.tokenize(doc)
        
        if p: print(f"\t* [TOKENS] {' '.join(tokens)}")
        
        if self.pos_tagging:
            # Apply pos (part-of-speech) tagging for better lemmatization
            tokens = nltk.pos_tag(tokens)
            lemmatized_tokens = []
            for t in tokens:
                lemma = self.wnl.lemmatize(t[0], self.get_wordnet_pos(t[1]))
                lemmatized_tokens.append(lemma)
            if p:
                print(f"\t* [LEMMA] {' '.join(lemmatized_tokens)}")
            return lemmatized_tokens
        else: 
            # Apply lemmatization without pos_tagging
            return [self.wnl.lemmatize(t) for t in tokens]

######################

#### MAIN METHODS ####
def apply_bow(articles, use_vectorizer, use_token_pattern, use_analyzer, use_lemmatization, use_pos_tagging, use_fix_contractions, use_stopwords, use_stopwords_signature, min_df, max_df) -> pd.DataFrame:
    """
    This function takes the raw articles data and returns a dataframe that contains the same data represented with the bag of words format
    
    The implementation is based on: https://datascientyst.com/create-a-bag-of-words-pandas-python/
    """
    stopwordlist = None
    if use_stopwords:
        stopwordlist = stopwords.words('english')
        if use_stopwords_signature:
            stopwordlist += ["shameful", "cadre", "dsl", "n3jxp", "chastity", "skepticism", "surrender", "intellect", "geb", "gordon", "pitt", "bank", "soon", "edu"]
        
    vectorizer = None
    if use_vectorizer == "Tfidf":
        vectorizer = TfidfVectorizer(
            tokenizer = LemmaTokenizer(use_pos_tagging, use_fix_contractions, token_regex=use_token_pattern) if use_lemmatization else None,
            token_pattern=use_token_pattern,
            analyzer=use_analyzer,
            stop_words=stopwordlist,
            max_df=max_df,
            min_df=min_df,
        )
    elif use_vectorizer == "Count" or use_vectorizer == "Count_binary":
        vectorizer = CountVectorizer(
            tokenizer = LemmaTokenizer(use_pos_tagging, use_fix_contractions) if use_lemmatization else None,
            token_pattern=use_token_pattern,
            analyzer=use_analyzer,
            stop_words=stopwordlist,
            max_df=max_df,
            min_df=min_df,
            binary=use_vectorizer=="Count_binary"
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
        model = AgglomerativeClustering(n_clusters=n_clusters, **model_args)
        
    elif algorithm == "Spectral":
        model = SpectralClustering(n_clusters=n_clusters, **model_args)
        
    else:
        print(f"ERROR: algorithm '{algorithm}' is not supported.")
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

def find_anomalies(data, random_seed=1, add_text=False):
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
    isof = IsolationForest(
        contamination=50/2164, 
        random_state=random_seed,
        n_estimators=200,
        max_samples='auto',
    )
    anomalies_predictions = isof.fit_predict(data)
    
    anomalies_template = pd.read_csv(ANOMALIES_PATH)
    
    anomalies_alg_path = os.path.join(PROCESSED_DATA_PATH, f"anomalies.csv")
    anomalies_df = data[anomalies_predictions == -1]
    anomalies_df = anomalies_df.reset_index()[['doc_id']]
    anomalies_template["doc_id"] = anomalies_df["doc_id"]
    
    if add_text:
        text_data = pd.read_csv(ARTICLES_PATH)
        anomalies_template = pd.merge(anomalies_template, text_data, "left", "doc_id")
    
    anomalies_template.to_csv(anomalies_alg_path, index=False)

######################


if __name__ == "__main__":
    ### FLAGS ###
    RANDOM_SEED = 1
    # Preprocessing
    SKIP_PREPROCESS = False # If true, should have "articles_preprocessed.csv" in processed_data
                            # NOTE: Skipping preprocessing could cause reproducability issues.
    PREPROCESS_FLAGS = {
        "clustering": {
            "use_vectorizer": "Tfidf",              # Either 'Count', 'Count_binary' or 'Tfidf'
            "use_token_pattern": r"(?u)\b\w\w+\b",  # Default from sklearns CountVectorizer token_pattern,
            "use_analyzer": "word",                 # Default for clustering words
            "use_lemmatization": True,
            "use_fix_contractions": True,           # Extends lemmatization
            "use_pos_tagging": True,                # Extends lemmatization
            "use_stopwords": True,
            "use_stopwords_signature": True,
            "min_df": 0.03,
            "max_df": 0.7 
        },
        "anomaly": {
            "use_vectorizer": "Count",
            "use_token_pattern": r"(?u)\b\w\w+\b",  # N/A when using character analyzer
            "use_analyzer": "char_wb",              # look at individual characters instead of words
            "use_lemmatization": False,             # N/A when using character analyzer
            "use_fix_contractions": False,          # "
            "use_pos_tagging": False,               # "
            "use_stopwords": False,                 # "
            "use_stopwords_signature": False,       # "
            "min_df": 1,
            "max_df": 0.4
        }
    }
    # Clustering
    SSE_CURVE = True    # Plot the SSE graph for all cluster counts (KMeans only)
    RUN_ALGORITHMS = {  # Specify which algorithms to run (with which arguments) and with which cluster count
        "KMeans": {"modelArgs": {"n_init": 10, "random_state": RANDOM_SEED}, "clusterCounts": [i for i in range(2,11)]}, 
        "Hierarchical": {"modelArgs": {}, "clusterCounts": [i for i in range(2,11)]}
    }
    PRINT_TERM_FILES = {
        "algo": [
            "KMeans",       # Comment these out to stop printing
            "Hierarchical",
        ],
        "clusters": [i for i in range(2,11)],
        "top_terms": 5,
        "print_frequency": False # Print the full map (with frequency data of the top terms), or only print the top terms
    }
    # Anomaly detection
    ANOMALY_IF = True   # Isolation Forest approach
    #############
    
    articles = pd.read_csv(ARTICLES_PATH)
    
    toDownload = []
    
    if PREPROCESS_FLAGS["clustering"]["use_lemmatization"] == True or PREPROCESS_FLAGS["anomaly"]["use_lemmatization"] == True:
        toDownload.append('wordnet')
    if PREPROCESS_FLAGS["clustering"]["use_pos_tagging"] == True or PREPROCESS_FLAGS["anomaly"]["use_pos_tagging"] == True:
        toDownload.append('averaged_perceptron_tagger_eng')
    if PREPROCESS_FLAGS["clustering"]["use_stopwords"] == True or PREPROCESS_FLAGS["anomaly"]["use_stopwords"] == True:
        toDownload.append('stopwords')
        
    if len(toDownload) > 0 and not SKIP_PREPROCESS:
        print(f"Downloading NLTK_DATA: {toDownload}")
        if NLTK_DATA_PATH not in nltk.data.path:
            nltk.data.path.append(NLTK_DATA_PATH)
        nltk.download(toDownload, download_dir=NLTK_DATA_PATH)

    ### Clustering ###
    if RUN_ALGORITHMS:
        ### Preprocess ###
        if not SKIP_PREPROCESS: 
            print("Preprocessing articles data...")
            articles = apply_bow(articles, **PREPROCESS_FLAGS["clustering"])
            print("Preprocessing DONE!")
                
            # Preprocessed data
            articles.to_csv(PROCESSED_DATA_PATH + "articles_preprocessed.csv",
                            float_format="%.20f") # Helps to not lose float precision for reproducability
        else:
            # Assume preprocessing is already done and do not re-do it, just load
            articles = pd.read_csv(PROCESSED_DATA_PATH + "articles_preprocessed.csv",index_col="doc_id")
            
        ### Clustering ###
        all_silhouette_scores = {}
        for algo in RUN_ALGORITHMS:
            algo_args = RUN_ALGORITHMS[algo]["modelArgs"]
            
            ## Make sure directory for the algorithm exists and clear it
            algo_path = os.path.join(PROCESSED_DATA_PATH, algo)
            if os.path.exists(algo_path):
                shutil.rmtree(algo_path)    # Clear all files from the previous run
            else:
                os.makedirs(algo_path)
                
            sse_scores = []
            silhouette_scores = {}
            
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
                silhouette_scores[cluster_count] = resultsDict["silhouette"]
                
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
                        cluster_file_count = len(cluster_df)
                        sorted_term_frequencies.insert(0, 'cluster_file_count', cluster_file_count)
                        sorted_term_frequencies.to_csv(specific_cluster_terms_filepath,index=False)
                        ####################
                        
                        clusterFile_path = os.path.join(algo_clusterCount_path, f"cluster_{int(label)}.csv")
                        cluster_df.to_csv(clusterFile_path, index_label='DOC_ID')   
                    
                    # Make sure to remove the added collumn again
                    articles = articles.drop(columns=['cluster_label'])
            
            silhouette_path = os.path.join(algo_path, "silhouette_scores.csv")
            silhouette_df = pd.Series(silhouette_scores).to_frame(name='silhouette_score')
            silhouette_df.to_csv(silhouette_path, index_label='cluster_count')
            all_silhouette_scores[algo] = silhouette_scores
            
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
                
                plot_path = os.path.join(algo_path, "delta_sse_graph.png")
                plt.figure()
                delta_sse_scores = [sse_scores[i-1] - sse_scores[i] for i in range(1, len(sse_scores))]    
                x_labels = [f"[{i-1},{i}]" for i in RUN_ALGORITHMS[algo]["clusterCounts"][1:]]           
                plt.bar(x_labels, delta_sse_scores, color='b', label='delta SSE')
                plt.title('delta SSE scores for KMeans')
                plt.xlabel('Number of clusters')
                plt.ylabel('Delta SSE')
                plt.grid(True)
                plt.savefig(plot_path)
        
        # Generate silhouette scores graph
        silhouette_graph_path = os.path.join(PROCESSED_DATA_PATH, "silhouette_scores.png")
        plt.figure()
        for key in all_silhouette_scores.keys():
            cluster_count_list = sorted(all_silhouette_scores[key].keys())
            score_list = [all_silhouette_scores[key][c] for c in cluster_count_list]
            plt.plot(cluster_count_list, score_list, marker='x', label=key)
        plt.title('Silhouette scores for all cluster counts per algorithm')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette score')
        plt.grid(True)
        plt.legend()
        plt.savefig(silhouette_graph_path)
        
        #########################
        
        for alg in PRINT_TERM_FILES["algo"]:
            print(f"{alg}:")
            for clus in PRINT_TERM_FILES["clusters"]:
                print(f"  * Run: {clus} clusters")
                read_path = os.path.join(PROCESSED_DATA_PATH, alg, f"{clus}_clusters")
                for f in os.listdir(read_path):
                    if f"_terms" in f:
                        f_path = os.path.join(read_path, f)
                        read_df = pd.read_csv(f_path)
                        read_map = {c: round(read_df[c][0], 1) for c in list(read_df.columns)[:PRINT_TERM_FILES["top_terms"]+1]}
                        if PRINT_TERM_FILES["print_frequency"]:
                            print(f"\t{f} top {PRINT_TERM_FILES['top_terms']} terms: {read_map}")
                        else:
                            print(f"\t{f} (Articles: {read_map.get('cluster_file_count', 0)})\ttop {PRINT_TERM_FILES['top_terms']} terms: {', '.join(list(read_map.keys())[1:])}")

    ### ANOMALY DETECTION ###
    if ANOMALY_IF == True:
        print("Starting anomaly detection...")
        if not SKIP_PREPROCESS: 
            articles = pd.read_csv(ARTICLES_PATH)
            print("Preprocessing articles data for anomaly detection...")
            articles = apply_bow(articles, **PREPROCESS_FLAGS["anomaly"])
            print("Preprocessing DONE!")
                
            # Preprocessed data
            articles.to_csv(PROCESSED_DATA_PATH + "articles_preprocessed_anomaly.csv",
                            float_format="%.20f") # Helps to not lose float precision for reproducability
        else:
            # Assume preprocessing is already done and do not re-do it, just load
            articles = pd.read_csv(PROCESSED_DATA_PATH + "articles_preprocessed_anomaly.csv",index_col="doc_id")
        
        find_anomalies(data=articles, random_seed=RANDOM_SEED)
        print("Anomaly detection DONE!")

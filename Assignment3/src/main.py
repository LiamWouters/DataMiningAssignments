## PROVIDED DATA ##
# articles.csv: doc_id,text
###################

## EVENTUAL PREDICTION ##
# anomalies.csv: anomaly,doc_id
# clusters.csv: doc_id,label
#########################

import pandas as pd
import matplotlib.pyplot as plt
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
    articles = apply_bow(
        articles,
        use_tfidf=PREPROCESS_TFIDFVECTORIZER,
        use_lemmatization=PREPROCESS_LEMMATIZATION,
        use_pos_tagging=PREPROCESS_LEMMATIZATION_POSTAGGING,
        use_stopwords=PREPROCESS_STOPWORDS,
        min_df=MIN_DF if PREPROCESS_MINDF else 1,
        max_df=MAX_DF if PREPROCESS_MAXDF else 1.0,
    )
        
    # Preprocessed data
    articles.to_csv(PROCESSED_DATA_PATH + "after_preprocessing.csv")
    
    ### Clustering ###

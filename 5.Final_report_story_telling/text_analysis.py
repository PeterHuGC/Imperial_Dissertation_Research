import pandas as pd

# 1. for LDA section
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from nltk.tokenize import sent_tokenize, word_tokenize

import matplotlib.pyplot as plt



from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import common_texts
from gensim.models import LdaMulticore, CoherenceModel # coherence to find the optimal number of topics

import matplotlib.colors as mcolors

# download nltk package
nltk.download('stopwords')
nltk.download('punkt')

stopword=set(stopwords.words('english'))

def data_cleaner(text, return_tokens = False):
    '''
    Cleans the data from special characters, urls, punctuation marks, extra spaces.
    Removes stopwords (Like if, it, the etc) and transforms the word in its native
    form using Porter Stemmer.
    '''
    text = str(text).lower() # lowercase the string
    text = re.sub('\[.*?\]', ' ', text) # replace punctuation with whitespaces.
    text = re.sub('https?://\S+|www\.\S+', ' ', text) # replacing urls with whitespaces.
    text = re.sub('<.*?>+', ' ', text) # removes special characters
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text) # removes punctuation
    text = re.sub('\r', ' ', text) # removes new line characters
    text = re.sub('\n', ' ', text) # removes new line characters
    text = re.sub('\w*\d\w*', ' ', text)
    #text = re.sub('–', ' ', text) # remove any additional characters we cannot remove 
    text = re.sub('[–£…»]', ' ', text) # remove any additional characters we cannot remove 
    text = text.split()

    # removing stopwords.
    text = [word for word in text if not word in stopword]

    # stemming.
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text]

    if return_tokens:

        # return relevant tokens here where needed
        return text

    #List to string.
    text = ' '.join(text)

    return text


# 2. for sentiment analysis 

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "ProsusAI/finbert"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def predict_sentiment(texts):
    # Encode texts
    encoded_input = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
    # Model prediction
    with torch.no_grad():
        outputs = model(**encoded_input)
    
    # given logits take the predictions for the most likely sentiment label
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1) 
    # represents the labels = ['positive', 'neutral', 'negative']
    labels = [1, 0 , -1] 
    
    predicted_labels = [labels[torch.argmax(prediction)] for prediction in predictions][0]
    
    return predicted_labels


def get_report_sentiments(reports_df : pd.DataFrame):
    """
    Get the report sentiments for dataframe that already contains 
    business overview (section 1) and risk overview (section 1A)
    """

    reports_df["risk_sentiment"] = reports_df["Text_1A"].apply(predict_sentiment)
    reports_df["business_overview_sentiment"] = reports_df["Text_1"].apply(predict_sentiment)

    return reports_df


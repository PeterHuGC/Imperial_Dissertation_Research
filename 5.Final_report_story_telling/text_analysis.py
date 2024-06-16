import pandas as pd
import numpy as np

# 1. for LDA section
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from nltk.tokenize import sent_tokenize, word_tokenize

import matplotlib.pyplot as plt

from collections import Counter
import gensim
from gensim.corpora.dictionary import Dictionary
from gensim.test.utils import common_texts
from gensim.models import LdaMulticore, CoherenceModel # coherence to find the optimal number of topics
from sklearn.manifold import TSNE


import matplotlib.colors as mcolors
from tqdm import tqdm

# download nltk package data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

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

    # list to string.
    text = ' '.join(text)

    return text

def clean_tokenise_risk_assessment_txt(sec_10k_df:pd.DataFrame):

    text_lda = sec_10k_df["Text_1A"]
    docs_tok = text_lda.apply(data_cleaner, return_tokens=True)
    sec_10k_df["Text_1A_data_cleaned"] = docs_tok

    return sec_10k_df


def select_optimal_LDA_topic_number(docs_tok, common_corpus, common_dictionary, min_topics = 2, max_topics = 4):

    model_list = []
    coherence_values = []

    for num_topics in tqdm(range(min_topics, max_topics + 1)):
        # we train the lda model - using the same code from the lda lab in uda
        lda_model = LdaMulticore(corpus=common_corpus,
                                id2word=common_dictionary,
                                num_topics=num_topics,
                                # workers=10,
                                passes=10)
        
        model_list.append(lda_model)
        
        # here we calculate the coherence score
        # recommended method for optimal topic on matheworks and stackoverflow
        # https://stackoverflow.com/questions/17421887/how-to-determine-the-number-of-topics-for-lda
        # https://uk.mathworks.com/help/textanalytics/ug/choose-number-of-topics-for-LDA-model.html
        coherence_model_lda = CoherenceModel(model=lda_model, texts=docs_tok, dictionary=common_dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        coherence_values.append(coherence_lda)

    # select the optimal model topics based on stackoverflow
    optimal_model_index = coherence_values.index(max(coherence_values))
    optimal_model = model_list[optimal_model_index]
    optimal_num_topics = min_topics + optimal_model_index

    print("Optimal Number of Topics:", optimal_num_topics)

    # plot out the topic details

    topics_range = range(min_topics, max_topics + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(topics_range, coherence_values, marker='o')
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.title("LDA Model Coherence Scores by Number of Topics")
    plt.xticks(topics_range)
    plt.grid(True)

    plt.show()

    return optimal_model, optimal_num_topics


def plot_topic_compositions(lda_model, common_dictionary, figsize=(16,10)):
    topics = lda_model.show_topics(formatted=False)
    counter = Counter(common_dictionary)

    out = []
    for i, topic in enumerate(topics):
        for word, weight in topic[1]:
            out.append([word, i , weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

    # Plotting Weights of Topic Keywords
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True)
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, label='Weights')
        ax.set_ylim(0, 0.030)
        ax.set_title('Topic: ' + str(i), color=cols[i])
        ax.tick_params(axis='y', left=True)
        ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
        ax.legend(loc='upper right')

    fig.tight_layout(w_pad=2)    
    fig.suptitle('Importance of Topic Keywords', fontsize=22, y=1.05)    
    plt.show()

def plot_dominant_topic_distributions(lda_model, common_corpus):

    doc_topic_matrix = [lda_model.get_document_topics(bow) for bow in common_corpus]

    # convert to document-topic matrix
    matrix = gensim.matutils.corpus2dense(doc_topic_matrix, num_terms=lda_model.num_topics)

    # get dominant topic for each document
    dominant_topic = np.argmax(matrix, axis=0)

    # t-SNE
    tsne_model = TSNE(n_components=2, perplexity=50, learning_rate=100, n_iter=2000, random_state=0)
    tsne_values = tsne_model.fit_transform(matrix.T)

    colors = plt.cm.tab10(np.linspace(0, 1, 6))

    plt.figure(figsize=(12, 6))
    for i, color in zip(range(lda_model.num_topics), colors):
        indices = dominant_topic == i
        plt.scatter(tsne_values[indices, 0], tsne_values[indices, 1], c=color, label=f'Topic {i}')

    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE visualization of LDA document-topic distribution')
    plt.legend(loc='best')

# 2. for sentiment analysis 

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "ProsusAI/finbert"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def predict_sentiment(texts):
    
    # encode and predict sentiments
    encoded_input = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
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


def plot_annual_report_sentiments(sec_10k_df, sentiment_col, years_lst):

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    # plot by year

    for i, year in enumerate(years_lst):

        yearly_data = sec_10k_df[sec_10k_df['report_year'] == year]
        
        # count sentiments
        sentiment_counts = yearly_data[sentiment_col].value_counts().reindex([-1, 0, 1], fill_value=0)
    
        axs[i].bar(sentiment_counts.index, sentiment_counts.values)
        axs[i].set_title(f'{sentiment_col.replace("_"," ").title()} Counts for {year}')
        axs[i].set_xlabel('Sentiment')
        axs[i].set_ylabel('Count')

    plt.tight_layout()
    plt.show()
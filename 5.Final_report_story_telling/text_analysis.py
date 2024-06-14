import pandas as pd


# for sentiment analysis 

from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "ProsusAI/finbert"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

import torch

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
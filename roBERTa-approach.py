import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

# f-String holding information about the model we want to use
MODEL = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"

""" This script will use a pretained RoBERTa model from Hugging Face,
it is a Bidirectional Encoder Representation from Transformers approach. 

The ML model used here has been pre-trained on ~198 Million tweets and
finetuned for sentiment analysis by CardiffNLP - a Cardiff University 
Research group.
https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment
https://arxiv.org/abs/2104.12250
"""

# Read data
df = pd.read_csv('./data-input/Reviews.csv')
df = df.head(1000)
 
# Setup pretrained tokenizer and stored model 
# weights from the address above
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

"""Example for investigating in interactive/jupyter"""
# example = df['Text'][50]
# encoded_example = tokenizer(example, return_tensors='pt')
# example_output = model(**encoded_example)
# # Now have a classifier object containing a tensor
# # Extract scores from object
# example_scores = example_output[0][0].detach().numpy()
# # apply softmax function, get array with [neg, neu, pos] scores
# example_scores = softmax(example_scores)
# scores_dict = {
#     'roberta_neg' : example_scores[0],
#     'roberta_neu' : example_scores[1],
#     'roberta_pos' : example_scores[2]
# }

# Simple function to get polarity scores from roberta model
def polarity_scores_roberta(example):
    encoded_txt = tokenizer(example, return_tensors='pt')
    output = model(**encoded_txt)
    # Now have a classifier object containing a tensor
    # Extract scores from object
    scores = output[0][0].detach().numpy()
    # apply softmax function, get array with [neg, neu, pos] scores
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

results = {}
# Include try-except to avoid runtime errors for text too long for model
# Quick and easy (dirty) fix while investigating
for i, row in df.iterrows():
    try:
        text = row['Text']
        item_id = row['Id']
        roberta_result = polarity_scores_roberta(text)
        results[item_id] = roberta_result
    except RuntimeError:
        print(f'Runtime error for item [{item_id}]')

roberta_results = pd.DataFrame(results).T
roberta_results = roberta_results.reset_index().rename(columns={'index': 'Id'})
roberta_results = roberta_results.merge(df, how='left')

# We now have a set of 3 scores similar to the Vader approach,
# Using a more sophisticated method/model
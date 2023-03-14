import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

# f-String holding information about the model we want to use
MODEL = f"cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Read data
df = pd.read_csv('./data-input/Reviews.csv')
df = df.head(500)

# Instantiate analyser object:
sia = SentimentIntensityAnalyzer()
# Pretrained tokenizer, stored model weights from the address above
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

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

        # Get VADER scores
        vader_result = sia.polarity_scores(text)
        # Need to rename & clarify the vader scores to differentiate them:
        vader_renamed = {}
        for key, value in vader_result.items():
            vader_renamed[f'vader_{key}'] = value
        
        # Get Roberta scores
        roberta_result = polarity_scores_roberta(text)
        
        #Combine the two sets of scores in to single dictionary
        combined = {**vader_renamed, **roberta_result}
        results[item_id] = combined
    except RuntimeError:
        print(f'Runtime error for item [{item_id}]')

results_df = pd.DataFrame(results).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')

# Plot to look at the differences in polarity vs review scores
# for the different approaches:

sns.pairplot(data = results_df,
            vars = ['vader_neg', 'vader_neu', 'vader_pos', 
                     'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue = 'Score',
            palette='tab10'
)
plt.show()
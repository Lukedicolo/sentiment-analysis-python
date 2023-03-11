import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns

from nltk.sentiment import SentimentIntensityAnalyzer

# NTLK Downloader if any parts are missing first time / in vEnv
#nltk.download()

"""
This script will use the NLTK SentimentIntensityAnalyzer to get polarity scores (Negative, Neutral, Positive)
scores of the text associated with the amazon reviews, using a VADER approach
VADER: Valence Aware Dictionary and sEntiment Reasoner (So should be called *VADSR*...)
    More laymans term is a 'bag of words' approach: remove stop words, score remaining words, combine scores.

Approach doesn't account for any relationships between words    
"""

# Read data
df = pd.read_csv('./data-input/Reviews.csv')
#df = df.head(1000)

# Instantiate analyser object:
sia = SentimentIntensityAnalyzer()

# Test /example cases of using the SIA
# print(sia.polarity_scores("I am incredibly happy today!"))
# print(sia.polarity_scores("This week has been miserable."))
# print(sia.polarity_scores(df['Text'][50]))

# Evaluate polarity scores of the entire dataset (text column)
# Dictionary to hold ID/scores
results = {}

for i, row in df.iterrows():
    text = row['Text']
    item_id = row['Id']
    results[item_id] = sia.polarity_scores(text)

# Build dataframe of this:
vaders = pd.DataFrame(results).T # Transform orientation
# Want this DF in suitable format to combine with original DF:
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
# Merge from the left, inserts new Sentiment scores to left of other data
vaders = vaders.merge(df, how='left')

""" An assumption we have is that products with higher review scores will also have
more positive review texts, this can be investigated graphically. """

## Plot out VADER results

# Compound score:
ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compound polarity score by Amazon review score')
plt.show()

# Individual scores:
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
sns.barplot(data=vaders, x = 'Score', y='neg', ax=axs[0])
sns.barplot(data=vaders, x = 'Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x = 'Score', y='pos', ax=axs[2])
axs[0].set_title('Negative')
axs[1].set_title('Neutral')
axs[2].set_title('Positive')
plt.tight_layout()
plt.show()

""" As expected, these plots show us that higher scoring products tend to have
more positive language present in their reviews according to their compound polarity,
further inspecting polarity scores present in the text, we see that negative language
decreases with score, neutral language is rouhgly equally present, and positive 
language polarity increases with review score.

Something extra to note is that this more simple approach would be unable to 
assess things like sarcasm, which may switch the meaning of words from negative
to positive. A more advanced approach may lead to more accurate representation.
"""
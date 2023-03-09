import pandas as pd
import matplotlib.pyplot as plt

import nltk
#nltk.download()

# Optionally: set a plotting style
plt.style.use('ggplot')

# First read in data
df = pd.read_csv('./data-input/Reviews.csv')
# df.shape() shows that there are over 500k rows of data.
df = df.head(1000) # For shorter runtime in dev TODO- remove this later


# Basic exploratory analysis:
Scores = df['Score'].value_counts().sort_index()
ax = Scores.plot(kind='bar', title='Review rating counts', figsize=(10, 6))
ax.set_xlabel('Review score (Stars)')
ax.set_ylabel('Count')
plt.show()
# plot shows bias towards higher reviews, slight exception with 1-star reviews.


# Basic NLTK 
example = df['Text'][50]
print(example)

# Tokenise the text to make it more CPU-interpretable
tokens = nltk.word_tokenize(example)
tokens[:10]

# Add part-of-speech values to the tokens
tagged = nltk.pos_tag(tokens)
tagged[:10]

# Group the tagged tokens into sentence 'chunks'
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()
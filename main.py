import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Optionally: set a plotting style
plt.style.use('ggplot')

# First read in data
df = pd.read_csv('./data-input/Reviews.csv')

# df.shape() shows that there are over 500k rows of data.
df = df.head(500) # For shorter runtime in dev TODO- remove this later

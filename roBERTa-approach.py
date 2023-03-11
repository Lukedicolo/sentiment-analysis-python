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

# Setup pretrained tokenizer and model from the address above
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


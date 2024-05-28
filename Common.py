import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.style.use('dark_background')
import pandas as pd
import plotly.express as px
from geopy.geocoders import Nominatim
from wordcloud import WordCloud
from collections import Counter
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from plotly.subplots import make_subplots
from statsmodels.formula.api import ols
from scipy import stats
import string
import nltk
from nltk.corpus import stopwords
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    
nltk.download('stopwords')

dataset = pd.read_csv('./data/Airplane_Crashes_and_Fatalities_Since_1908.csv')
dataset.describe().T.style.bar(
    subset=['mean'],
    color='#606ff2').background_gradient(
    subset=['std'], cmap='PuBu').background_gradient(subset=['50%'], cmap='PuBu')

dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset['Year'] = pd.DatetimeIndex(dataset['Date']).year
dataset['Month'] = pd.DatetimeIndex(dataset['Date']).month

dataset['Manufacturer'] = dataset['Type'].apply(lambda x: x.split(' ')[0] if type(x) is str else 'Others')

dataset.isnull().values.any()

text = ' '.join(dataset['Summary'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

words = text.lower().split()

# Remove punctuation during word splitting
words = [word.strip(string.punctuation) for word in words]

# Filter out non-alphabetic tokens and stopwords
filtered_words = [word for word in words if word.isalpha() and word not in stopwords.words('english')]

# Count the frequency of each word
word_freq = Counter(filtered_words)

# Sort the words by frequency
sorted_word_freq = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True))

# Get the top N most frequent words and their counts
top_n = 20
top_words = list(sorted_word_freq.keys())[:top_n]
word_counts = list(sorted_word_freq.values())[:top_n]
fig = px.bar(x=top_words, y=word_counts, template='plotly_dark', title=f'Top {top_n} Most Frequent Words in Summary')
fig.show()
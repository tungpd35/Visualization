import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.style.use('dark_background')
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
from geopy.geocoders import Nominatim
from wordcloud import WordCloud
from collections import Counter
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats
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

fig1 = px.histogram(data_frame=dataset, x='Fatalities', title='Distribution of Fatalities', template='plotly_dark')
fig1.show()
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stopwords_indonesian = stopwords.words('indonesian')

st.title('Sentiment Analysis App Titipku')
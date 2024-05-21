import streamlit as st
from wordcloud import WordCloud
# import os
import pandas as pd
import seaborn as sns
# import collections
sns.set_theme(color_codes=True)
import numpy as np
import matplotlib.pyplot as plt
import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification 
TF_ENABLE_ONEDNN_OPTS=0

st.title('Sentiment Analysis App Titipku')

upload_file = st.file_uploader("Upload a CSV File", type="csv")
added_stopwords = st.text_input("Enter Add Stopword (comma-separated)")
added_stopword_list = [word.strip() for word in added_stopwords.split(",")] if added_stopwords else []

if upload_file is not None:
    df = pd.read_csv(upload_file)
    st.subheader('Preview Dataset')
    st.dataframe(df)
    object_columns = df.select_dtypes(include="object").columns
    target_variable = st.selectbox("Choose a column for Sentiment Analysis:", object_columns)

    if st.button('Analyze'):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        stop_factory = StopWordRemoverFactory()
        more_stopword = ['byk', 'dg', 'yg', 'tdk', 'gak', 'sy', 'ga', 'nya', 'lg', 'jd', 'bs', 'dr']
        stopwords_indonesian = stop_factory.get_stop_words() + more_stopword

        st.success(f"Sentiment Analysis performed on column: {target_variable}")

        st.write(f"Selected {target_variable} Column:")
        st.dataframe(df[[target_variable]])

        new_df = df.copy()

        def clean_review(review):
            review = emoji.replace_emoji(review, replace='')
            review = re.sub(r'\@w+|\#', '', review)
            review = re.sub(r'[^A-Za-z\s]', '', review)
            review = re.sub(r'\s+', ' ', review).strip()
            review = review.lower()
            tokens = word_tokenize(review)
            tokens = [word for word in tokens if word not in stopwords_indonesian]
            lemmatized_words = [stemmer.stem(word) for word in tokens]

            return ' '.join(lemmatized_words)

        pretrained= "mdhugol/indonesia-bert-sentiment-classification"
        model = AutoModelForSequenceClassification.from_pretrained(pretrained)
        tokenizer = AutoTokenizer.from_pretrained(pretrained)
        sentiment_classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        label_index = {'LABEL_0': 'Positive', 'LABEL_1': 'Neutral', 'LABEL_2': 'Negative'}
        
        new_df['cleaned_text'] = df.apply(lambda x: clean_review(x[target_variable]), axis=1)

        data = (
            new_df
            .assign(sentiment = lambda x:x['cleaned_text'].apply(lambda s : sentiment_classifier(s)))
            .assign(
                label = lambda x : x['sentiment'].apply(lambda s:label_index[s[0]['label']]),
                score = lambda x : x['sentiment'].apply(lambda s:s[0]['score'])
            )
        )
        # Display the results
        st.subheader("Sentiment Analysis Results:")
        st.dataframe(data)

        sentiment_counts = data['label'].value_counts()

        # Plot a bar chart using seaborn
        st.set_option('deprecation.showPyplotGlobalUse', False)
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment Label')
        plt.ylabel('Count')
        st.pyplot()

        data['label'].value_counts().sort_index() \
        .plot(kind='pie', autopct='%1.1f%%', title='Percentage Label Reviews')
        st.pyplot()

        # def generate_wordcloud(text):
        #     wordcloud = WordCloud(width=800, height=400, background_color ='white', stopwords = None, min_font_size = 10).generate(text)
        #     plt.figure(figsize=(10, 5))
        #     plt.imshow(wordcloud, interpolation='bilinear')
        #     plt.axis("off")
        #     st.pyplot()
        
        # st.subheader('Wordcloud All Review')
        # text = data['cleaned_text'].str.cat(sep=' ')
        # generate_wordcloud(text)

        # st.subheader('Wordcloud Negative Review')
        # text_negative = data[data['label']=='Negative']
        # text_negative = text_negative['cleaned_text'].str.cat(sep=' ')
        # generate_wordcloud(text_negative)

        # st.subheader('Wordcloud Positive Review')
        # text_positive = data[data['label']=='Positive']
        # text_positive = text_positive['cleaned_text'].str.cat(sep=' ')
        # generate_wordcloud(text_positive)

        negative_reviews = data[data['label']=='Negative']
        negative_reviews_sorted = negative_reviews.sort_values(by='score', ascending=False).head(10)
        st.write('Top 10 Negative Review')
        st.dataframe(negative_reviews_sorted)

        positive_reviews = data[data['label']=='Positive']
        positive_reviews_sorted = positive_reviews.sort_values(by='score', ascending=False).head(10)
        st.write('Top 10 Positive Review')
        st.dataframe(positive_reviews_sorted)

        sentiment_text = {
            "Positive": "",
            "Neutral": "",
            "Negative": ""
        }

        # Loop through each sentiment label
        for label in sentiment_counts.index:
            # Filter data for the current sentiment
            selected_data = data[data['label'] == label]

            # Include custom stopwords back into the cleaned text before concatenation
            selected_data['cleaned_text'] = selected_data['cleaned_text'].apply(lambda x: ' '.join([word for word in x.split()]))

            # Concatenate cleaned text from the selected data (now including custom stopwords)
            sentiment_text[label] = ' '.join(selected_data['cleaned_text'].astype(str))

        # Concatenate cleaned text for each sentiment
        positive_text = ' '.join([word for word in data[data['label'] == 'Positive']['cleaned_text'].apply(lambda x: ' '.join([w for w in x.split()]))])
        neutral_text = ' '.join([word for word in data[data['label'] == 'Neutral']['cleaned_text'].apply(lambda x: ' '.join([w for w in x.split()]))])
        negative_text = ' '.join([word for word in data[data['label'] == 'Negative']['cleaned_text'].apply(lambda x: ' '.join([w for w in x.split()]))])

        # Generate WordCloud for positive sentiment
        positive_wordcloud = WordCloud(
            min_font_size=3, max_words=200, width=800, height=400,
            colormap='viridis', background_color='white'
        ).generate(positive_text)

        # Save the WordCloud image with a filename
        positive_wordcloud_filename = "wordcloud_positive.png"
        positive_wordcloud.to_file(positive_wordcloud_filename)

        # Display the saved WordCloud image using Streamlit
        st.subheader("WordCloud for Positive Sentiment")
        st.image(positive_wordcloud_filename)

        # with open(positive_wordcloud_filename, "rb") as file:
        #     btn = st.download_button(
        #             label="Download Wordcloud ",
        #             data=file,
        #             file_name="wordcloud_positive.png",
        #             mime="image/png"
        #         )

        # # Bigrams Positive Sentiment
        # words1 = positive_text.split()
        # # Get bigrams
        # bigrams = list(zip(words1, words1[1:]))

        # # Count bigrams
        # bigram_counts = collections.Counter(bigrams)

        # # Get top 10 bigram counts
        # top_bigrams = dict(bigram_counts.most_common(10))

        # # Create bar chart
        # plt.figure(figsize=(10, 7))
        # plt.bar(range(len(top_bigrams)), list(top_bigrams.values()), align='center')
        # plt.xticks(range(len(top_bigrams)), list(top_bigrams.keys()), rotation=90)
        # plt.xlabel('Bigram Words')
        # plt.ylabel('Count')
        # plt.title(f"Top 10 Bigram for Positive Sentiment")
        # # Save the entire plot as a PNG
        # plt.tight_layout()
        # plt.savefig("bigram_positive.png")
        # st.subheader("Bigram for Positive Sentiment")
        # st.image("bigram_positive.png")

        negative_wordcloud = WordCloud(
            min_font_size=3, max_words=200, width=800, height=400,
            colormap='viridis', background_color='white'
        ).generate(negative_text)

        # Save the WordCloud image with a filename
        negative_wordcloud_filename = "wordcloud_negative.png"
        negative_wordcloud.to_file(negative_wordcloud_filename)

        # Display the saved WordCloud image using Streamlit
        st.subheader("WordCloud for Negative Sentiment")
        st.image(negative_wordcloud_filename)

        # # Bigrams negative Sentiment
        # words2 = negative_text.split()
        # # Get bigrams
        # bigrams = list(zip(words2, words2[1:]))

        # # Count bigrams
        # bigram_counts = collections.Counter(bigrams)

        # # Get top 10 bigram counts
        # top_bigrams = dict(bigram_counts.most_common(10))

        # # Create bar chart
        # plt.figure(figsize=(10, 7))
        # plt.bar(range(len(top_bigrams)), list(top_bigrams.values()), align='center')
        # plt.xticks(range(len(top_bigrams)), list(top_bigrams.keys()), rotation=90)
        # plt.xlabel('Bigram Words')
        # plt.ylabel('Count')
        # plt.title(f"Top 10 Bigram for Negative Sentiment")
        # # Save the entire plot as a PNG
        # plt.tight_layout()
        # plt.savefig("bigram_negative.png")
        # st.subheader("Bigram for Negative Sentiment")
        # st.image("bigram_negative.png")

        neutral_wordcloud = WordCloud(
            min_font_size=3, max_words=200, width=800, height=400,
            colormap='viridis', background_color='white'
        ).generate(neutral_text)

        # Save the WordCloud image with a filename
        neutral_wordcloud_filename = "wordcloud_neutral.png"
        neutral_wordcloud.to_file(neutral_wordcloud_filename)

        # Display the saved WordCloud image using Streamlit
        st.subheader("WordCloud for Neutral Sentiment")
        st.image(neutral_wordcloud_filename)

        # # Bigrams neutral Sentiment
        # words3 = neutral_text.split()
        # # Get bigrams
        # bigrams = list(zip(words3, words3[1:]))

        # # Count bigrams
        # bigram_counts = collections.Counter(bigrams)

        # # Get top 10 bigram counts
        # top_bigrams = dict(bigram_counts.most_common(10))

        # # Create bar chart
        # plt.figure(figsize=(10, 7))
        # plt.bar(range(len(top_bigrams)), list(top_bigrams.values()), align='center')
        # plt.xticks(range(len(top_bigrams)), list(top_bigrams.keys()), rotation=90)
        # plt.xlabel('Bigram Words')
        # plt.ylabel('Count')
        # plt.title(f"Top 10 Bigram for Neutral Sentiment")
        # # Save the entire plot as a PNG
        # plt.tight_layout()
        # plt.savefig("bigram_neutral.png")
        # st.subheader("Bigram for Neutral Sentiment")
        # st.image("bigram_neutral.png")
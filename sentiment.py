import streamlit as st
from wordcloud import WordCloud
import pandas as pd
import seaborn as sns
sns.set_theme(color_codes=True)
import matplotlib.pyplot as plt
import re
import emoji
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification 
TF_ENABLE_ONEDNN_OPTS=0

nltk.download('punkt')
nltk.download('wordnet')

st.set_page_config(page_title="Sentiment Analysis")
st.title('Sentiment Analysis App Titipku')

url = "https://scrapping-app-review.streamlit.app/"
st.write("Follow this [link](%s) to scrapping dataset" % url)

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

        # Bert Model
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

        # Plot a distribution
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

        # Generate WordCloud for negative sentiment
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

        # Generate WordCloud for neutral sentiment
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
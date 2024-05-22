from google_play_scraper import app, Sort, reviews_all
from app_store_scraper import AppStore
import pandas as pd
import numpy as np
import json, os, uuid
import streamlit as st

st.title("App Review Scrapper")

st.write("The app link is highlighted in yellow")
st.image('link-playstore.png')
playstore = st.text_input("Enter App Link in Playstore")

st.write("The app link is highlighted in green, and ID in blue")
st.image("link-appstore.png")
appstore = st.text_input("Enter App Link in Appstore")
appstoreid = st.text_input("Enter ID App in Appstore")

if st.button('Scrapping'):
    g_reviews = reviews_all(
            # "com.titipku.alpha",
            playstore,
            sleep_milliseconds=0,
            lang='id',
            country='id',
            sort=Sort.NEWEST,
        )

    g_df = pd.DataFrame(np.array(g_reviews),columns=['review'])
    g_df2 = g_df.join(pd.DataFrame(g_df.pop('review').tolist()))

    g_df2.drop(columns={'userImage', 'reviewCreatedVersion'},inplace = True)
    g_df2.rename(columns= {'score': 'rating','userName': 'user_name', 'reviewId': 'review_id', 'content': 'review_description', 'at': 'review_date', 'replyContent': 'developer_response', 'repliedAt': 'developer_response_date', 'thumbsUpCount': 'thumbs_up'},inplace = True)
    g_df2.insert(loc=0, column='source', value='Google Play')
    g_df2.insert(loc=3, column='review_title', value=None)
    g_df2['laguage_code'] = 'in'
    g_df2['country_code'] = 'id'

    st.subheader("Google Playstore Review")
    st.dataframe(g_df2)

    count1 = len(g_df2)

    st.write(f'Counts of Google Playstore Review : {count1}')

    a_reviews = AppStore('id', appstore, appstoreid)
    a_reviews.review()

    a_df = pd.DataFrame(np.array(a_reviews.reviews),columns=['review'])
    a_df2 = a_df.join(pd.DataFrame(a_df.pop('review').tolist()))

    a_df2.drop(columns={'isEdited'},inplace = True)
    a_df2.insert(loc=0, column='source', value='App Store')
    a_df2['developer_response_date'] = None
    a_df2['thumbs_up'] = None
    a_df2['laguage_code'] = 'in'
    a_df2['country_code'] = 'id'
    a_df2.insert(loc=1, column='review_id', value=[uuid.uuid4() for _ in range(len(a_df2.index))])
    a_df2.rename(columns= {'review': 'review_description','userName': 'user_name', 'date': 'review_date','title': 'review_title', 'developerResponse': 'developer_response'},inplace = True)
    a_df2 = a_df2.where(pd.notnull(a_df2), None)

    st.subheader("Apple Appstore Review")
    st.dataframe(a_df2)

    count2 = len(a_df2)
    st.write(f'Counts of Apple Appstore Review : {count2}')

    result = pd.concat([g_df2,a_df2]).reset_index()
    final_result = result[['source','user_name','review_description','review_date','rating']]

    st.subheader("Final Review")
    st.dataframe(final_result)

    count = len(final_result)
    st.write(f'Counts of Total Review : {count}')

    datas = final_result.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download data as CSV",
        data=datas,
        file_name="review_app_titipku.csv",
        mime="text/csv",
    )
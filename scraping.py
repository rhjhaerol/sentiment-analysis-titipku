from google_play_scraper import app, Sort, reviews_all
from app_store_scraper import AppStore
import pandas as pd
import numpy as np
import json, os, uuid
import streamlit as st

st.set_page_config(page_title="Scraper App Review")
st.title("App Review Scraper")

urlg = "https://play.google.com/"
st.write("You can visit the Play Store at this [link](%s) and search for the app that you want to scrape." %urlg)
st.write("Example: The app link is highlighted in yellow")
st.image('link-playstore.png')
playstore = st.text_input("Enter App Link in Playstore")

if st.button('Scraping Google Playstore Review'):
    if playstore:
        playstore_scrape = reviews_all(
            playstore,
            sleep_milliseconds=0,
            lang='id',
            country='id',
            sort=Sort.NEWEST)

        df_playstore = pd.json_normalize(playstore_scrape)

        df_playstore.drop(columns={'userImage', 'reviewCreatedVersion'},inplace = True)
        df_playstore.rename(columns= {'score': 'rating','userName': 'user_name', 'reviewId': 'review_id', 'content': 'review_description', 'at': 'review_date', 'replyContent': 'developer_response', 'repliedAt': 'developer_response_date', 'thumbsUpCount': 'thumbs_up'},inplace = True)
        df_playstore.insert(loc=0, column='source', value='Google Play')

        st.subheader("Google Playstore Review")
        st.dataframe(df_playstore)

        countg = len(df_playstore)

        st.write(f'Counts of Google Playstore Review : {countg}')
        st.success('Data scraped successfully')

        data1 = df_playstore[['source','user_name', 'review_description','review_date','rating']]

        datasg = data1.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download data as CSV",
            data=datasg,
            file_name="review_app_playstore.csv",
            mime="text/csv",
        )
    else: 
        st.write("Please enter the App Link.")

st.markdown("***")

urla = "https://www.apple.com/app-store/"
st.markdown("You can visit the App Store at this [link](%s) and search for the app that you want to scrape." %urla)
st.write("Example: The app link is highlighted in green, and the ID in blue.")
st.image("link-appstore.png")

appstore = st.text_input("Enter App Link in Appstore")
appstoreid = st.text_input("Enter ID App in Appstore")

if st.button('Scraping App Store Review'):
    if appstore and appstoreid:
        appstore_scrape = AppStore('id', appstore, appstoreid)
        appstore_scrape.review()

        df_appstore = pd.json_normalize(appstore_scrape.reviews)

        df_appstore.insert(loc=0, column='source', value='App Store')
        df_appstore['developer_response_date'] = df_appstore['developerResponse.modified']
        df_appstore['developer_response'] = df_appstore['developerResponse.body']
        df_appstore['thumbs_up'] = None
        df_appstore.insert(loc=1, column='review_id', value=[uuid.uuid4() for _ in range(len(df_appstore.index))])
        df_appstore.rename(columns= {'review': 'review_description','userName': 'user_name', 'date': 'review_date','title': 'review_title', 'developerResponse': 'developer_response'},inplace = True)
        df_appstore = df_appstore.where(pd.notnull(df_appstore), None)
        df_appstore.drop(columns={'isEdited', 'developerResponse.modified', 'developerResponse.body', 'developerResponse.id' },inplace = True)

        st.subheader("Apple Appstore Review")
        st.dataframe(df_appstore)

        counta = len(df_appstore)
        st.write(f'Counts of Apple Appstore Review : {counta}')
        st.success('Data scraped successfully')

        data2 = df_appstore[['source','user_name', 'review_description','review_date','rating']]

        datasa = data2.to_csv(index=False).encode("utf-8")
        
        st.download_button(
            label="Download data as CSV",
            data=datasa,
            file_name="review_app_appstore.csv",
            mime="text/csv",
        )
    else: 
        st.write("Please enter both the App Link and the App ID.")

st.markdown("***")

if st.button('Scraping All Review'):
    if playstore and appstore and appstoreid:
        playstore_scrape = reviews_all(
            playstore,
            sleep_milliseconds=0,
            lang='id',
            country='id',
            sort=Sort.NEWEST)

        df_playstore = pd.json_normalize(playstore_scrape)

        df_playstore.drop(columns={'userImage', 'reviewCreatedVersion'},inplace = True)
        df_playstore.rename(columns= {'score': 'rating','userName': 'user_name', 'reviewId': 'review_id', 'content': 'review_description', 'at': 'review_date', 'replyContent': 'developer_response', 'repliedAt': 'developer_response_date', 'thumbsUpCount': 'thumbs_up'},inplace = True)
        df_playstore.insert(loc=0, column='source', value='Google Play')

        st.subheader("Google Playstore Review")
        st.dataframe(df_playstore)

        count1 = len(df_playstore)

        st.write(f'Counts of Google Playstore Review : {count1}')

        appstore_scrape = AppStore('id', appstore, appstoreid)
        appstore_scrape.review()

        df_appstore = pd.json_normalize(appstore_scrape.reviews)

        df_appstore.insert(loc=0, column='source', value='App Store')
        df_appstore['developer_response_date'] = df_appstore['developerResponse.modified']
        df_appstore['developer_response'] = df_appstore['developerResponse.body']
        df_appstore['thumbs_up'] = None
        df_appstore.insert(loc=1, column='review_id', value=[uuid.uuid4() for _ in range(len(df_appstore.index))])
        df_appstore.rename(columns= {'review': 'review_description','userName': 'user_name', 'date': 'review_date','title': 'review_title', 'developerResponse': 'developer_response'},inplace = True)
        df_appstore = df_appstore.where(pd.notnull(df_appstore), None)
        df_appstore.drop(columns={'isEdited', 'developerResponse.modified', 'developerResponse.body', 'developerResponse.id' },inplace = True)


        st.subheader("Apple Appstore Review")
        st.dataframe(df_appstore)

        count2 = len(df_appstore)
        st.write(f'Counts of Apple Appstore Review : {count2}')

        final_result = pd.concat([df_playstore,df_appstore]).reset_index()

        st.subheader("Final Review")
        st.dataframe(final_result)

        count = len(final_result)
        st.write(f'Counts of Total Review : {count}')
        st.success('Data scraped successfully')

        data3 = final_result[['source','user_name', 'review_description','review_date','rating']]

        datas = data3.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download data as CSV",
            data=datas,
            file_name="review_app_all.csv",
            mime="text/csv",
        )
    else:
        st.write('Please enter third the App Link.')
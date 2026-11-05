import numpy as np
import pandas as pd
import googleapiclient.discovery
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, GlobalMaxPooling1D, Concatenate, Input
import os
import json
from tensorflow.keras.models import Model
import datetime

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')

API_KEY = 'INSERTAPIKEY'
youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=API_KEY)


def getRandomVideos(saveFile=True, filePath='randomVideos2.json', numVideos=20):
    # Check if the file already exists
    if os.path.exists(filePath):
        # If the file exists, load and return the data from the file
        with open(filePath, 'r') as file:
            videosData = json.load(file)
        return videosData

    # If the file doesn't exist, fetch new random videos
    videosData = []

    # Search for videos with the specified criteria
    videos = youtube.search().list(
        part='snippet',
        type='video',
        relevanceLanguage='en',  # English videos
        maxResults=numVideos,
        videoDuration='any',   # Any duration
        videoDefinition='high',  # High definition
        safeSearch='strict',
    ).execute()


    videoIds = [item['id']['videoId'] for item in videos['items']]
    # Get video details (title, views) for the selected video IDs
    for videoId in videoIds:
        response = youtube.videos().list(
            part='snippet,statistics',
            id=videoId
        ).execute()
        
        snippet = response['items'][0]['snippet']
        statistics = response['items'][0]['statistics']

        title = snippet['title']
        description = snippet.get('description', '')
        channel_title = snippet['channelTitle']
        channel_id = snippet['channelId']
        subscriber_count = get_channel_subscriber_count(channel_id)
        category_id = snippet['categoryId']
        publication_date = snippet['publishedAt']
        
        langMatch = True
        if ('defaultAudioLanguage' in snippet and snippet['defaultAudioLanguage'] != 'en') or ('defaultLanguage' in snippet and snippet['defaultLanguage'] != 'en'):
            langMatch = False
        
        
        # Ensure the video has more than 1,000 views, exists, and is in english
        if (langMatch and 'viewCount' in statistics and int(statistics['viewCount']) > 1000):
            videosData.append({
                'videoId': videoId,
                'title': title,
                'description': description,
                'channelTitle': channel_title,
                'channelId': channel_id,
                'subscriberCount': subscriber_count,
                'categoryId': category_id,
                'publishedAt': publication_date,
                'views': int(statistics['viewCount'])
            })
        
        

    # Save the data to a file
    if saveFile:
        with open(filePath, 'w') as file:
            json.dump(videosData, file)

    return videosData

def getPopularVideos(saveFile=True, filePath='popularVideos2.json'):
    # Check if the file already exists
    if os.path.exists(filePath):
        # If the file exists, load and return the data from the file
        with open(filePath, 'r') as file:
            videosData = json.load(file)
        return videosData

    # If the file doesn't exist, fetch new random videos
    videosData = []
    
    popularVids = youtube.videos().list(
        part='snippet,statistics',
        chart='mostPopular',
        regionCode='US',  # Adjust the region code as needed
        maxResults=50
    ).execute()

    for video in popularVids['items']:
        snippet = video['snippet']
        statistics = video['statistics']

        title = snippet['title']
        description = snippet.get('description', '')
        channel_title = snippet['channelTitle']
        channel_id = snippet['channelId']
        subscriber_count = get_channel_subscriber_count(channel_id)
        category_id = snippet['categoryId']
        publication_date = snippet['publishedAt']

        lang_match = True
        if ('defaultAudioLanguage' in snippet and snippet['defaultAudioLanguage'] != 'en') or ('defaultLanguage' in snippet and snippet['defaultLanguage'] != 'en'):
            lang_match = False

        # Ensure the video has more than 1,000 views and exists
        if ('viewCount' in statistics and int(statistics['viewCount']) > 1000 and lang_match):
            videosData.append({
                'videoId': video['id'],
                'title': title,
                'description': description,
                'channelTitle': channel_title,
                'channelId': channel_id,
                'subscriberCount': subscriber_count,
                'categoryId': category_id,
                'publishedAt': publication_date,
                'views': int(statistics['viewCount'])
            })




    # Save the data to a file
    if saveFile:
        with open(filePath, 'w') as file:
            json.dump(videosData, file)

    return videosData


def getCategoryVideos(saveFile=True, filePath='categoryVideos2.json', numVideos=20):
    # Check if the file already exists
    if os.path.exists(filePath):
        # If the file exists, load and return the data from the file
        with open(filePath, 'r') as file:
            videosData = json.load(file)
        return videosData

    # If the file doesn't exist, fetch new random videos
    videosData = []

    # Search for videos with the specified criteria
    videos = youtube.search().list(
        part='snippet',
        type='video',
        relevanceLanguage='en',  # English videos
        maxResults=numVideos,
        category_id=20,#gaming
        videoDuration='any',   # Any duration
        videoDefinition='high',  # High definition
        safeSearch='strict',
    ).execute()


    videoIds = [item['id']['videoId'] for item in videos['items']]
    # Get video details (title, views) for the selected video IDs
    for videoId in videoIds:
        response = youtube.videos().list(
            part='snippet,statistics',
            id=videoId
        ).execute()
        
        snippet = response['items'][0]['snippet']
        statistics = response['items'][0]['statistics']

        title = snippet['title']
        description = snippet.get('description', '')
        channel_title = snippet['channelTitle']
        channel_id = snippet['channelId']
        subscriber_count = get_channel_subscriber_count(channel_id)
        category_id = snippet['categoryId']
        publication_date = snippet['publishedAt']
        
        langMatch = True
        if ('defaultAudioLanguage' in snippet and snippet['defaultAudioLanguage'] != 'en') or ('defaultLanguage' in snippet and snippet['defaultLanguage'] != 'en'):
            langMatch = False
        
        
        # Ensure the video has more than 1,000 views, exists, and is in english
        if (langMatch and 'viewCount' in statistics and int(statistics['viewCount']) > 1000):
            videosData.append({
                'videoId': videoId,
                'title': title,
                'description': description,
                'channelTitle': channel_title,
                'channelId': channel_id,
                'subscriberCount': subscriber_count,
                'categoryId': category_id,
                'publishedAt': publication_date,
                'views': int(statistics['viewCount'])
            })
        
        

    # Save the data to a file
    if saveFile:
        with open(filePath, 'w') as file:
            json.dump(videosData, file)

    return videosData


def getChannelsVideos(saveFile=True, filePath='channelsVideos3.json'):
    # Check if the file already exists
    if os.path.exists(filePath):
        # If the file exists, load and return the data from the file
        with open(filePath, 'r') as file:
            videosData = json.load(file)
        return videosData

    # If the file doesn't exist, fetch videos from specific channels
    videosData = []

    # channelIds = ['UCgRQHK8Ttr1j9xCEpCAlgbQ', 'UCHnyfMqiRRG1u-2MsSQLbXA', 'UCGfUuxBzB8E30XjCjOvji2w', 'UCBJycsmduvYEL83R_U4JriQ', 'UCvK4bOhULCpmLabd2pDMtnA',
    #                'UCc9CjaAjsMMvaSghZB7-Kog', 'UCX6OQ3DkcsbYNE6H8uQQuVA', 'UCQJWtTnAHhEG5w4uN0udnUQ', 'UCL6JmiMXKoXS6bpP1D3bk8g', 'UCZB6V9fUov0Mx_us3MWWILg']
    
    # channelIds = ['UClyGlKOhDUooPJFy4v_mqPg', 'UCw1SQ6QRRtfAhrN_cjkrOgA', 'UCTkXRDQl0luXxVQrRQvWS6w', 'UCLXo7UDZvByw2ixzpQCufnA', 
    #               'UCAuk798iHprjTtwlClkFxMA', 'UCimiUgDLbi6P17BdaCZpVbg', 'UCDogdKl7t7NHzQ95aEwkdMw', 'UC2C_jShtL725hvbm1arSV9w', 
    #               'UCftwRNsjfRo08xYE31tkiyw', 'UCyps-v4WNjWDnYRKmZ4BUGw', 'UCsEukrAd64fqA7FjwkmZ_Dw', 'UCq6VFHwMzcMXbuKyG7SQYIg', 'UCTSRIY3GLFYIpkR2QwyeklA']
    
    # channelIds = ['UCgRQHK8Ttr1j9xCEpCAlgbQ', 'UCHnyfMqiRRG1u-2MsSQLbXA', 'UCGfUuxBzB8E30XjCjOvji2w', 'UCBJycsmduvYEL83R_U4JriQ', 
    #               'UCvK4bOhULCpmLabd2pDMtnA', 'UCc9CjaAjsMMvaSghZB7-Kog', 'UCX6OQ3DkcsbYNE6H8uQQuVA', 'UCQJWtTnAHhEG5w4uN0udnUQ', 
    #               'UCL6JmiMXKoXS6bpP1D3bk8g', 'UCZB6V9fUov0Mx_us3MWWILg', 'UClyGlKOhDUooPJFy4v_mqPg', 'UCw1SQ6QRRtfAhrN_cjkrOgA', 
    #               'UCTkXRDQl0luXxVQrRQvWS6w', 'UCLXo7UDZvByw2ixzpQCufnA', 'UCAuk798iHprjTtwlClkFxMA', 'UCimiUgDLbi6P17BdaCZpVbg', 
    #               'UCDogdKl7t7NHzQ95aEwkdMw', 'UC2C_jShtL725hvbm1arSV9w', 'UCftwRNsjfRo08xYE31tkiyw', 'UCyps-v4WNjWDnYRKmZ4BUGw', 
    #               'UCsEukrAd64fqA7FjwkmZ_Dw', 'UCq6VFHwMzcMXbuKyG7SQYIg', 'UCTSRIY3GLFYIpkR2QwyeklA']
    
    channelIds = ['UCgRQHK8Ttr1j9xCEpCAlgbQ', 'UCHnyfMqiRRG1u-2MsSQLbXA', 'UCGfUuxBzB8E30XjCjOvji2w', 'UCBJycsmduvYEL83R_U4JriQ', 
                  'UCvK4bOhULCpmLabd2pDMtnA', 'UCc9CjaAjsMMvaSghZB7-Kog', 'UCX6OQ3DkcsbYNE6H8uQQuVA', 'UCQJWtTnAHhEG5w4uN0udnUQ', 
                  'UCL6JmiMXKoXS6bpP1D3bk8g', 'UCZB6V9fUov0Mx_us3MWWILg', 'UClyGlKOhDUooPJFy4v_mqPg', 'UCw1SQ6QRRtfAhrN_cjkrOgA', 
                  'UCTkXRDQl0luXxVQrRQvWS6w', 'UCLXo7UDZvByw2ixzpQCufnA', 'UCAuk798iHprjTtwlClkFxMA', 'UCimiUgDLbi6P17BdaCZpVbg', 
                  'UCDogdKl7t7NHzQ95aEwkdMw', 'UC2C_jShtL725hvbm1arSV9w', 'UCftwRNsjfRo08xYE31tkiyw', 'UCyps-v4WNjWDnYRKmZ4BUGw']
    
    
    for channelID in channelIds:
        # Get the latest videos from the channel
        channelResponse = youtube.search().list(
            channelId=channelID,
            part='id',
            type='video',
            maxResults=20
        ).execute()

        videoIds = [item['id']['videoId'] for item in channelResponse['items']]

        # Get video details (title, views) for the selected video IDs
        for video_id in videoIds:
            response = youtube.videos().list(
                part='snippet,statistics',
                id=video_id
            ).execute()
            snippet = response['items'][0]['snippet']
            statistics = response['items'][0]['statistics']

            title = snippet['title']
            description = snippet.get('description', '')
            channel_title = snippet['channelTitle']
            channel_id = snippet['channelId']
            subscriber_count = get_channel_subscriber_count(channel_id)
            category_id = snippet['categoryId']
            publication_date = snippet['publishedAt']

            lang_match = True
            if ('defaultAudioLanguage' in snippet and snippet['defaultAudioLanguage'] != 'en') or ('defaultLanguage' in snippet and snippet['defaultLanguage'] != 'en'):
                lang_match = False

            # Ensure the video has more than 1,000 views and exists
            if ('viewCount' in statistics and int(statistics['viewCount']) > 1000 and lang_match):
                videosData.append({
                    'videoId': video_id,
                    'title': title,
                    'description': description,
                    'channelTitle': channel_title,
                    'channelId': channel_id,
                    'subscriberCount': subscriber_count,
                    'categoryId': category_id,
                    'publishedAt': publication_date,
                    'views': int(statistics['viewCount'])
                })

    # Save the data to a file
    if saveFile:
        with open(filePath, 'w') as file:
            json.dump(videosData, file)

    return videosData


def getPopularChannelsVideos(saveFile=True, filePath='popularChannelsVideos4.json'):
    # Check if the file already exists
    if os.path.exists(filePath):
        # If the file exists, load and return the data from the file
        with open(filePath, 'r') as file:
            videosData = json.load(file)
        return videosData

    # If the file doesn't exist, fetch videos from channels with videos on trending
    videosData = []

    trendingList = youtube.videos().list(
        part='snippet',
        chart='mostPopular',
        regionCode='US',  # Adjust the region code as needed
        maxResults=20
    ).execute()

    trendingChannelIds = [item['snippet']['channelId'] for item in trendingList['items']]

    for channelID in trendingChannelIds:
        # Get the latest videos from the channel
        channelResponse = youtube.search().list(
            channelId=channelID,
            part='id',
            type='video',
            maxResults=20
        ).execute()

        videoIds = [item['id']['videoId'] for item in channelResponse['items']]

        # Get video details (title, views) for the selected video IDs
        for videoId in videoIds:
            response = youtube.videos().list(
                part='snippet,statistics',
                id=videoId
            ).execute()
            snippet = response['items'][0]['snippet']
            statistics = response['items'][0]['statistics']

            title = snippet['title']
            description = snippet.get('description', '')
            channel_title = snippet['channelTitle']
            channel_id = snippet['channelId']
            subscriber_count = get_channel_subscriber_count(channel_id)
            category_id = snippet['categoryId']
            publication_date = snippet['publishedAt']

            lang_match = True
            if ('defaultAudioLanguage' in snippet and snippet['defaultAudioLanguage'] != 'en') or ('defaultLanguage' in snippet and snippet['defaultLanguage'] != 'en'):
                lang_match = False

            # Ensure the video has more than 1,000 views and exists
            if ('viewCount' in statistics and int(statistics['viewCount']) > 1000 and lang_match):
                videosData.append({
                    'videoId': videoId,
                    'title': title,
                    'description': description,
                    'channelTitle': channel_title,
                    'channelId': channel_id,
                    'subscriberCount': subscriber_count,
                    'categoryId': category_id,
                    'publishedAt': publication_date,
                    'views': int(statistics['viewCount'])
                })

    # Save the data to a file
    if saveFile:
        with open(filePath, 'w') as file:
            json.dump(videosData, file)

    return videosData


def getRandomChannelsVideos(saveFile=True, filePath='randomChannelsVideos8.json', numChannels=20, numVideosPerChannel=20):
    # Check if the file already exists
    if os.path.exists(filePath):
        # If the file exists, load and return the data from the file
        with open(filePath, 'r') as file:
            videosData = json.load(file)
        return videosData

    # If the file doesn't exist, fetch videos from specific channels
    videosData = []

    # Search for videos with the specified criteria
    searchResponse = youtube.search().list(
        part='snippet',
        type='video',
        relevanceLanguage='en',  # English videos
        maxResults=numChannels,
        videoDuration='any',   # Any duration
        videoDefinition='high',  # High definition
        safeSearch='strict',
    ).execute()

    channelIds = [item['snippet']['channelId'] for item in searchResponse['items']]

    for channelID in channelIds:
        # Get the latest videos from the channel
        channelResponse = youtube.search().list(
            channelId=channelID,
            part='id',
            type='video',
            maxResults=numVideosPerChannel
        ).execute()

        videoIds = [item['id']['videoId'] for item in channelResponse['items']]

        # Get video details (title, views) for the selected video IDs
        for videoId in videoIds:
            response = youtube.videos().list(
                part='snippet,statistics',
                id=videoId
            ).execute()
            snippet = response['items'][0]['snippet']
            statistics = response['items'][0]['statistics']

            title = snippet['title']
            description = snippet.get('description', '')
            channel_title = snippet['channelTitle']
            channel_id = snippet['channelId']
            subscriber_count = get_channel_subscriber_count(channel_id)
            category_id = snippet['categoryId']
            publication_date = snippet['publishedAt']

            lang_match = True
            if ('defaultAudioLanguage' in snippet and snippet['defaultAudioLanguage'] != 'en') or ('defaultLanguage' in snippet and snippet['defaultLanguage'] != 'en'):
                lang_match = False

            # Ensure the video has more than 1,000 views and exists
            if ('viewCount' in statistics and int(statistics['viewCount']) > 1000 and lang_match):
                videosData.append({
                    'videoId': videoId,
                    'title': title,
                    'description': description,
                    'channelTitle': channel_title,
                    'channelId': channel_id,
                    'subscriberCount': subscriber_count,
                    'categoryId': category_id,
                    'publishedAt': publication_date,
                    'views': int(statistics['viewCount'])
                })

    # Save the data to a file
    if saveFile:
        with open(filePath, 'w') as file:
            json.dump(videosData, file)

    return videosData


def combineDatasets(saveFile=True):
    # Check if the file already exists
    dataFiles = ['combinedVideos', 'allVideos', 'popularChannelsVideos', 'channelsVideos', 'randomChannelsVideos', 'categoryVideos']
    # dataFiles = ['popularChannelsVideos', 'channelsVideos', 'randomChannelsVideos', 'categoryVideos']
    
    videosData = []
    
    for filePath in dataFiles:
        exists = True
        i = 2
        while exists and i < 10:
            if i != 1:
                if i == 2:
                    filePath += str(i)
                else:
                    filePath = filePath[:len(filePath)-1]
                    filePath += str(i)
            print(filePath)
            if os.path.exists(filePath+'.json'):
                # If the file exists, load and return the data from the file
                with open(filePath+'.json', 'r') as file:
                    videosData.append(json.load(file))
                    i += 1
            else:
                exists = False
            
    
    # Combine the lists of sets into one list with unique videoIds
    combinedList = []
    uniqueVideoIds = set()

    for videoList in videosData:
        for video in videoList:
            videoId = video["videoId"]
            if videoId not in uniqueVideoIds:
                uniqueVideoIds.add(videoId)
                combinedList.append(video)



    # Save the data to a file
    if saveFile:
        with open('combinedVideos3.json', 'w') as file:
            json.dump(combinedList, file)

    return combinedList

def getAllVideos(saveFile=True, filePath='allVideos2.json', numChannels=20, numVideosPerChannel=15):
    # Check if the file already exists
    if os.path.exists(filePath):
        # If the file exists, load and return the data from the file
        with open(filePath, 'r') as file:
            videosData = json.load(file)
        return videosData

    # If the file doesn't exist, fetch videos from specific channels
    videosData = []

    # Search for videos with the specified criteria
    searchResponse = youtube.search().list(
        part='snippet',
        type='video',
        relevanceLanguage='en',  # English videos
        maxResults=numChannels,
        videoDuration='any',   # Any duration
        videoDefinition='high',  # High definition
        safeSearch='strict',
    ).execute()

    randomChannelIds = [item['snippet']['channelId'] for item in searchResponse['items']]

    # specChannelIds = ['UCgRQHK8Ttr1j9xCEpCAlgbQ', 'UCHnyfMqiRRG1u-2MsSQLbXA', 'UCGfUuxBzB8E30XjCjOvji2w', 'UCBJycsmduvYEL83R_U4JriQ', 'UCvK4bOhULCpmLabd2pDMtnA',
    #                   'UCc9CjaAjsMMvaSghZB7-Kog', 'UCX6OQ3DkcsbYNE6H8uQQuVA', 'UCQJWtTnAHhEG5w4uN0udnUQ', 'UCL6JmiMXKoXS6bpP1D3bk8g', 'UCZB6V9fUov0Mx_us3MWWILg']

    specChannelIds = ['UCgRQHK8Ttr1j9xCEpCAlgbQ', 'UCHnyfMqiRRG1u-2MsSQLbXA', 'UCGfUuxBzB8E30XjCjOvji2w', 'UCBJycsmduvYEL83R_U4JriQ', 
                  'UCvK4bOhULCpmLabd2pDMtnA', 'UCc9CjaAjsMMvaSghZB7-Kog', 'UCX6OQ3DkcsbYNE6H8uQQuVA', 'UCQJWtTnAHhEG5w4uN0udnUQ', 
                  'UCL6JmiMXKoXS6bpP1D3bk8g', 'UCZB6V9fUov0Mx_us3MWWILg', 'UClyGlKOhDUooPJFy4v_mqPg', 'UCw1SQ6QRRtfAhrN_cjkrOgA', 
                  'UCTkXRDQl0luXxVQrRQvWS6w', 'UCLXo7UDZvByw2ixzpQCufnA', 'UCAuk798iHprjTtwlClkFxMA', 'UCimiUgDLbi6P17BdaCZpVbg', 
                  'UCDogdKl7t7NHzQ95aEwkdMw', 'UC2C_jShtL725hvbm1arSV9w', 'UCftwRNsjfRo08xYE31tkiyw', 'UCyps-v4WNjWDnYRKmZ4BUGw', 
                  'UCsEukrAd64fqA7FjwkmZ_Dw', 'UCq6VFHwMzcMXbuKyG7SQYIg', 'UCTSRIY3GLFYIpkR2QwyeklA']


    trendingList = youtube.videos().list(
        part='snippet',
        chart='mostPopular',
        regionCode='US',  # Adjust the region code as needed
        maxResults=20
    ).execute()

    trendingChannelIds = [item['snippet']['channelId'] for item in trendingList['items']]

    channelIds = list(set(randomChannelIds + specChannelIds + trendingChannelIds))

    for channelID in channelIds:
        # Get the latest videos from the channel
        channelResponse = youtube.search().list(
            channelId=channelID,
            part='id',
            type='video',
            maxResults=numVideosPerChannel
        ).execute()

        videoIds = [item['id']['videoId'] for item in channelResponse['items']]

        # Get video details (title, views) for the selected video IDs
        for videoId in videoIds:
            response = youtube.videos().list(
                part='snippet,statistics',
                id=videoId
            ).execute()
            snippet = response['items'][0]['snippet']
            statistics = response['items'][0]['statistics']

            title = snippet['title']
            description = snippet.get('description', '')
            channel_title = snippet['channelTitle']
            channel_id = snippet['channelId']
            subscriber_count = get_channel_subscriber_count(channel_id)
            category_id = snippet['categoryId']
            publication_date = snippet['publishedAt']

            lang_match = True
            if ('defaultAudioLanguage' in snippet and snippet['defaultAudioLanguage'] != 'en') or ('defaultLanguage' in snippet and snippet['defaultLanguage'] != 'en'):
                lang_match = False

            # Ensure the video has more than 1,000 views and exists
            if ('viewCount' in statistics and int(statistics['viewCount']) > 1000 and lang_match):
                videosData.append({
                    'videoId': videoId,
                    'title': title,
                    'description': description,
                    'channelTitle': channel_title,
                    'channelId': channel_id,
                    'subscriberCount': subscriber_count,
                    'categoryId': category_id,
                    'publishedAt': publication_date,
                    'views': int(statistics['viewCount'])
                })

    # Save the data to a file
    if saveFile:
        with open(filePath, 'w') as file:
            json.dump(videosData, file)

    return videosData






def get_channel_subscriber_count(channelId):
    """
    Get the subscriber count for a YouTube channel.

    Args:
        channel_id (str): The ID of the YouTube channel.

    Returns:
        int: The subscriber count of the channel.
    """
    try:
        channelResponse = youtube.channels().list(
            part='statistics',
            id=channelId
        ).execute()

        subscriberCount = int(channelResponse['items'][0]['statistics']['subscriberCount'])
        return subscriberCount

    except Exception as e:
        print(f"Error fetching subscriber count for channel {channelId}: {e}")
        return 0  # Return 0 in case of an error


# Preprocessing and cleaning the text data
def preprocessText(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stop words
    stopWords2 = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
    tokens = [token for token in tokens if token not in stopWords2]
    
    return ' '.join(tokens)


# Creates a model based only on title using neural network
def neuralTitleModel(df):
        
    #PART 2
    df['processed_title'] = df['title'].apply(preprocessText)
    print(df)

    # Tokenize titles
    max_words = 100000  # Adjust based on your dataset
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['processed_title'])
    sequences = tokenizer.texts_to_sequences(df['processed_title'])

    # Pad sequences
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')


    #PART 4
    # Split the dataset into training and testing sets
    X = padded_sequences  # Features
    y = df['views']  # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    #PART 5
    # Build a neural network model
    embedding_dim = 300  # You can adjust this based on your dataset and experiment
    lstm_units = 128  # You can adjust this based on your dataset and experiment

    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
    model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=True)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Display the model summary
    model.summary()

    #PART 6
    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluate the model on the test set
    loss = model.evaluate(X_test, y_test)
    print(f'Mean Squared Error on Test Set: {loss}')

    #PART 7
    # Make predictions on new data
    def predict_view_count(title):
        processed_title = preprocessText(title)
        sequence = tokenizer.texts_to_sequences([processed_title])
        padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
        # scaled_sequence = scaler.transform(padded_sequence)
        # prediction = model.predict(scaled_sequence)
        prediction = model.predict(padded_sequence)
        return prediction[0][0]

    # Example usage
    new_title = "I Survived 100 Days in Canada"
    predicted_views = predict_view_count(new_title)
    print(f'Predicted Views for "{new_title}": {predicted_views}')
    new_title = "$1 VS $1,000 Water"
    predicted_views = predict_view_count(new_title)
    print(f'Predicted Views for "{new_title}": {predicted_views}')
    
    
    return predicted_views


# Creates a model based on title and subscriber count using neural network
def neuralTitleSubscriberModel(df):
    # Part 2: Preprocessing
    df['processed_title'] = df['title'].apply(preprocessText)

    # Tokenize titles
    max_words = 100000  # Adjust based on your dataset
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['processed_title'])
    sequences = tokenizer.texts_to_sequences(df['processed_title'])
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

    # Part 2.1: Add subscriber count as a feature
    scaler = StandardScaler()
    subscriber_count_scaled = scaler.fit_transform(df[['subscriberCount']])

    # Combine title and subscriber count features
    X = np.concatenate([padded_sequences, subscriber_count_scaled], axis=1)

    y = df['views']  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Part 3: Build a neural network model
    embedding_dim = 300
    lstm_units = 128

    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len + 1))  # +1 for subscriber count
    model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=True)))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='linear'))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Display the model summary
    model.summary()

    # Part 4: Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Part 5: Evaluate the model on the test set
    loss = model.evaluate(X_test, y_test)
    print(f'Mean Squared Error on Test Set: {loss}')

    # Part 6: Make predictions on new data
    def predict_view_count(title, subscriber_count):
        processed_title = preprocessText(title)
        sequence = tokenizer.texts_to_sequences([processed_title])
        padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
        subscriber_count_scaled = scaler.transform([[subscriber_count]])
        # Combine title and subscriber count features
        features = np.concatenate([padded_sequence, subscriber_count_scaled], axis=1)
        prediction = model.predict(features)
        return prediction[0][0]

    # Example usage
    new_title = "I Survived 100 Days in Canada"
    subscriber_count = 1000000
    predicted_views = predict_view_count(new_title, subscriber_count)
    print(f'Predicted Views for "{new_title}": {predicted_views}')

    new_title = "$1 VS $1,000 Water"
    subscriber_count = 500000
    predicted_views = predict_view_count(new_title, subscriber_count)
    print(f'Predicted Views for "{new_title}": {predicted_views}')

    return predicted_views


# New function to train the model based on title, subscribers, and days since publication
def neuralTSDateModel(df):
    # Part 2: Preprocessing
    df['processed_title'] = df['title'].apply(preprocessText)


    # Tokenize titles
    max_words = 100000  # Adjust based on your dataset
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['processed_title'])
    sequences = tokenizer.texts_to_sequences(df['processed_title'])
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

    # Add subscriber count as a feature
    scaler_subscriber = StandardScaler()
    subscriber_count_scaled = scaler_subscriber.fit_transform(df[['subscriberCount']])

    # Add days since publication as a feature
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    df['daysSincePublication'] = (pd.to_datetime('today').tz_localize('UTC') - df['publishedAt']).dt.days
    scaler_days_since_publication = StandardScaler()
    days_since_publication_scaled = scaler_days_since_publication.fit_transform(df[['daysSincePublication']])

    # Combine title, subscriber count, and days since publication features
    X_title = padded_sequences
    X_subscriber = subscriber_count_scaled
    X_days_since_publication = days_since_publication_scaled

    # Define inputs using Keras Functional API
    input_title = Input(shape=(max_len,), name='title_input')
    input_subscriber = Input(shape=(1,), name='subscriber_input')
    input_days_since_publication = Input(shape=(1,), name='days_since_publication_input')

    # Embedding layer for title
    embedding_dim = 300
    title_embedding = Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len)(input_title)
    title_lstm = Bidirectional(LSTM(units=128, return_sequences=True))(title_embedding)
    title_pooled = GlobalMaxPooling1D()(title_lstm)

    # Fully connected layer for subscriber count
    subscriber_dense = Dense(units=64, activation='relu')(input_subscriber)

    # Fully connected layer for days since publication
    days_dense = Dense(units=64, activation='relu')(input_days_since_publication)

    # Concatenate the outputs from different branches
    merged = Concatenate()([title_pooled, subscriber_dense, days_dense])

    # Final fully connected layers
    merged_dense = Dense(units=128, activation='relu')(merged)
    merged_dropout = Dropout(0.5)(merged_dense)
    output_layer = Dense(units=1, activation='linear')(merged_dropout)

    # Build the model using the Functional API
    model = Model(inputs=[input_title, input_subscriber, input_days_since_publication], outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Display the model summary
    model.summary()

    y = df['views']  # Target variable
    # Split each input separately
    X_title_train, X_title_test, X_subscriber_train, X_subscriber_test, X_days_train, X_days_test, y_train, y_test = train_test_split(
        X_title, X_subscriber, X_days_since_publication, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit([X_title_train, X_subscriber_train, X_days_train], y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluate the model on the test set
    loss = model.evaluate([X_title_test, X_subscriber_test, X_days_test], y_test)
    print(f'Mean Squared Error on Test Set: {loss}')

        
    # Make predictions on new data
    def predict_view_count(title, subscriber_count, days_since_publication):
        processed_title = preprocessText(title)
        sequence = tokenizer.texts_to_sequences([processed_title])
        padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')
        subscriber_count_scaled = scaler_subscriber.transform([[subscriber_count]])
        days_since_publication_scaled = scaler_days_since_publication.transform([[days_since_publication]])
        
        # Prepare input for prediction
        X_pred = [padded_sequence, subscriber_count_scaled, days_since_publication_scaled]

        prediction = model.predict(X_pred)
        return prediction[0][0]

    # Example usage
    published_at = pd.to_datetime('2023-01-01')  # Assuming the videos were published at the start of the year
    days_since_publication = (pd.to_datetime('today').tz_localize('UTC') - published_at.tz_localize('UTC')).days
    new_title = "I Survived 100 Days in Canada"
    subscriber_count = 1000000
    # days_since_publication = (datetime.(2023, 12, 7) - datetime(2023, 1, 1)).days  # Example days since publication
    predicted_views = predict_view_count(new_title, subscriber_count, days_since_publication)
    print(f'Predicted Views for "{new_title}": {predicted_views}')

    new_title = "$1 VS $1,000 Water"
    subscriber_count = 500000
    # days_since_publication = (datetime.now() - datetime(2023, 1, 1)).days  # Example days since publication
    predicted_views = predict_view_count(new_title, subscriber_count, days_since_publication)
    print(f'Predicted Views for "{new_title}": {predicted_views}')

    return predicted_views



# New function to train the model based on title, subscribers, days since publication, and description
def neuralTSDDescriptionModel(df):
    # Part 2: Preprocessing
    df['processed_title'] = df['title'].apply(preprocessText)
    df['processed_description'] = df['description'].apply(preprocessText)


    # Tokenize titles and descriptions
    max_words_title = 100000  # Adjust based on your dataset
    max_words_description = 100000  # Adjust based on your dataset

    tokenizer_title = Tokenizer(num_words=max_words_title, oov_token="<OOV>")
    tokenizer_description = Tokenizer(num_words=max_words_description, oov_token="<OOV>")

    tokenizer_title.fit_on_texts(df['processed_title'])
    tokenizer_description.fit_on_texts(df['processed_description'])

    sequences_title = tokenizer_title.texts_to_sequences(df['processed_title'])
    sequences_description = tokenizer_description.texts_to_sequences(df['processed_description'])

    max_len_title = max(len(seq) for seq in sequences_title)
    max_len_description = max(len(seq) for seq in sequences_description)

    padded_sequences_title = pad_sequences(sequences_title, maxlen=max_len_title, padding='post')
    padded_sequences_description = pad_sequences(sequences_description, maxlen=max_len_description, padding='post')

    # Add subscriber count as a feature
    scaler_subscriber = StandardScaler()
    subscriber_count_scaled = scaler_subscriber.fit_transform(df[['subscriberCount']])

    # Add days since publication as a feature
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    df['daysSincePublication'] = (pd.to_datetime('today').tz_localize('UTC') - df['publishedAt']).dt.days
    scaler_days_since_publication = StandardScaler()
    days_since_publication_scaled = scaler_days_since_publication.fit_transform(df[['daysSincePublication']])

    # Combine title, description, subscriber count, and days since publication features
    X_title = padded_sequences_title
    X_description = padded_sequences_description
    X_subscriber = subscriber_count_scaled
    X_days_since_publication = days_since_publication_scaled

    # Define inputs using Keras Functional API
    input_title = Input(shape=(max_len_title,), name='title_input')
    input_description = Input(shape=(max_len_description,), name='description_input')
    input_subscriber = Input(shape=(1,), name='subscriber_input')
    input_days_since_publication = Input(shape=(1,), name='days_since_publication_input')

    # Embedding layer for title and description
    embedding_dim = 300
    title_embedding = Embedding(input_dim=max_words_title, output_dim=embedding_dim, input_length=max_len_title)(input_title)
    description_embedding = Embedding(input_dim=max_words_description, output_dim=embedding_dim, input_length=max_len_description)(input_description)

    title_lstm = Bidirectional(LSTM(units=128, return_sequences=True))(title_embedding)
    description_lstm = Bidirectional(LSTM(units=128, return_sequences=True))(description_embedding)

    title_pooled = GlobalMaxPooling1D()(title_lstm)
    description_pooled = GlobalMaxPooling1D()(description_lstm)

    # Fully connected layer for subscriber count
    subscriber_dense = Dense(units=64, activation='relu')(input_subscriber)

    # Fully connected layer for days since publication
    days_dense = Dense(units=64, activation='relu')(input_days_since_publication)

    # Concatenate the outputs from different branches
    merged = Concatenate()([title_pooled, description_pooled, subscriber_dense, days_dense])

    # Final fully connected layers
    merged_dense = Dense(units=128, activation='relu')(merged)
    merged_dropout = Dropout(0.5)(merged_dense)
    output_layer = Dense(units=1, activation='linear')(merged_dropout)

    # Build the model using the Functional API
    model = Model(inputs=[input_title, input_description, input_subscriber, input_days_since_publication], outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Display the model summary
    model.summary()

    y = df['views']  # Target variable
    # Split each input separately
    X_title_train, X_title_test, X_description_train, X_description_test, X_subscriber_train, X_subscriber_test, X_days_train, X_days_test, y_train, y_test = train_test_split(
        X_title, X_description, X_subscriber, X_days_since_publication, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit([X_title_train, X_description_train, X_subscriber_train, X_days_train], y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluate the model on the test set
    loss = model.evaluate([X_title_test, X_description_test, X_subscriber_test, X_days_test], y_test)
    print(f'Mean Squared Error on Test Set: {loss}')

    # Make predictions on new data
    def predict_view_count(title, description, subscriber_count, days_since_publication):
        processed_title = preprocessText(title)
        processed_description = preprocessText(description)
        sequence_title = tokenizer_title.texts_to_sequences([processed_title])
        sequence_description = tokenizer_description.texts_to_sequences([processed_description])

        padded_sequence_title = pad_sequences(sequence_title, maxlen=max_len_title, padding='post')
        padded_sequence_description = pad_sequences(sequence_description, maxlen=max_len_description, padding='post')

        subscriber_count_scaled = scaler_subscriber.transform([[subscriber_count]])
        days_since_publication_scaled = scaler_days_since_publication.transform([[days_since_publication]])

        # Prepare input for prediction
        X_pred = [padded_sequence_title, padded_sequence_description, subscriber_count_scaled, days_since_publication_scaled]

        prediction = model.predict(X_pred)
        return prediction[0][0]

    # Example usage
    published_at = pd.to_datetime('2023-01-01')  # Assuming the videos were published at the start of the year
    days_since_publication = (pd.to_datetime('today').tz_localize('UTC') - published_at.tz_localize('UTC')).days
    new_title = "I Survived 100 Days in Canada"
    new_description = "This is an amazing journey!"
    subscriber_count = 1000000
    predicted_views = predict_view_count(new_title, new_description, subscriber_count, days_since_publication)
    print(f'Predicted Views for "{new_title}": {predicted_views}')

    new_title = "$1 VS $1,000 Water"
    new_description = "Which one is better? Let's find out!"
    subscriber_count = 500000
    predicted_views = predict_view_count(new_title, new_description, subscriber_count, days_since_publication)
    print(f'Predicted Views for "{new_title}": {predicted_views}')

    return predicted_views


# New function to train the model based on title, subscribers, days since publication, description, and channel title
def neuralTSDDChannelModel(df):
    # Part 2: Preprocessing
    df['processed_title'] = df['title'].apply(preprocessText)
    df['processed_description'] = df['description'].apply(preprocessText)
    df['processed_channel_title'] = df['channelTitle'].apply(preprocessText)


    # Tokenize titles, descriptions, and channel titles
    max_words_title = 100000  # Adjust based on your dataset
    max_words_description = 100000  # Adjust based on your dataset
    max_words_channel_title = 100000  # Adjust based on your dataset

    tokenizer_title = Tokenizer(num_words=max_words_title, oov_token="<OOV>")
    tokenizer_description = Tokenizer(num_words=max_words_description, oov_token="<OOV>")
    tokenizer_channel_title = Tokenizer(num_words=max_words_channel_title, oov_token="<OOV>")

    tokenizer_title.fit_on_texts(df['processed_title'])
    tokenizer_description.fit_on_texts(df['processed_description'])
    tokenizer_channel_title.fit_on_texts(df['processed_channel_title'])

    sequences_title = tokenizer_title.texts_to_sequences(df['processed_title'])
    sequences_description = tokenizer_description.texts_to_sequences(df['processed_description'])
    sequences_channel_title = tokenizer_channel_title.texts_to_sequences(df['processed_channel_title'])

    max_len_title = max(len(seq) for seq in sequences_title)
    max_len_description = max(len(seq) for seq in sequences_description)
    max_len_channel_title = max(len(seq) for seq in sequences_channel_title)

    padded_sequences_title = pad_sequences(sequences_title, maxlen=max_len_title, padding='post')
    padded_sequences_description = pad_sequences(sequences_description, maxlen=max_len_description, padding='post')
    padded_sequences_channel_title = pad_sequences(sequences_channel_title, maxlen=max_len_channel_title, padding='post')

    # Add subscriber count as a feature
    scaler_subscriber = StandardScaler()
    subscriber_count_scaled = scaler_subscriber.fit_transform(df[['subscriberCount']])

    # Add days since publication as a feature
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    df['daysSincePublication'] = (pd.to_datetime('today').tz_localize('UTC') - df['publishedAt']).dt.days
    scaler_days_since_publication = StandardScaler()
    days_since_publication_scaled = scaler_days_since_publication.fit_transform(df[['daysSincePublication']])

    # Combine title, description, subscriber count, days since publication, and channel title features
    X_title = padded_sequences_title
    X_description = padded_sequences_description
    X_channel_title = padded_sequences_channel_title
    X_subscriber = subscriber_count_scaled
    X_days_since_publication = days_since_publication_scaled

    # Define inputs using Keras Functional API
    input_title = Input(shape=(max_len_title,), name='title_input')
    input_description = Input(shape=(max_len_description,), name='description_input')
    input_channel_title = Input(shape=(max_len_channel_title,), name='channel_title_input')
    input_subscriber = Input(shape=(1,), name='subscriber_input')
    input_days_since_publication = Input(shape=(1,), name='days_since_publication_input')

    # Embedding layer for title, description, and channel title
    embedding_dim = 300
    title_embedding = Embedding(input_dim=max_words_title, output_dim=embedding_dim, input_length=max_len_title)(input_title)
    description_embedding = Embedding(input_dim=max_words_description, output_dim=embedding_dim, input_length=max_len_description)(input_description)
    channel_title_embedding = Embedding(input_dim=max_words_channel_title, output_dim=embedding_dim, input_length=max_len_channel_title)(input_channel_title)

    title_lstm = Bidirectional(LSTM(units=128, return_sequences=True))(title_embedding)
    description_lstm = Bidirectional(LSTM(units=128, return_sequences=True))(description_embedding)
    channel_title_lstm = Bidirectional(LSTM(units=128, return_sequences=True))(channel_title_embedding)

    title_pooled = GlobalMaxPooling1D()(title_lstm)
    description_pooled = GlobalMaxPooling1D()(description_lstm)
    channel_title_pooled = GlobalMaxPooling1D()(channel_title_lstm)

    # Fully connected layer for subscriber count
    subscriber_dense = Dense(units=64, activation='relu')(input_subscriber)

    # Fully connected layer for days since publication
    days_dense = Dense(units=64, activation='relu')(input_days_since_publication)

    # Concatenate the outputs from different branches
    merged = Concatenate()([title_pooled, description_pooled, channel_title_pooled, subscriber_dense, days_dense])

    # Final fully connected layers
    merged_dense = Dense(units=128, activation='relu')(merged)
    merged_dropout = Dropout(0.5)(merged_dense)
    output_layer = Dense(units=1, activation='linear')(merged_dropout)

    # Build the model using the Functional API
    model = Model(inputs=[input_title, input_description, input_channel_title, input_subscriber, input_days_since_publication], outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Display the model summary
    model.summary()

    y = df['views']  # Target variable
    # Split each input separately
    X_title_train, X_title_test, X_description_train, X_description_test, X_channel_title_train, X_channel_title_test, X_subscriber_train, X_subscriber_test, X_days_train, X_days_test, y_train, y_test = train_test_split(
        X_title, X_description, X_channel_title, X_subscriber, X_days_since_publication, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit([X_title_train, X_description_train, X_channel_title_train, X_subscriber_train, X_days_train], y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluate the model on the test set
    loss = model.evaluate([X_title_test, X_description_test, X_channel_title_test, X_subscriber_test, X_days_test], y_test)
    print(f'Mean Squared Error on Test Set: {loss}')

    # Make predictions on new data
    def predict_view_count(title, description, channel_title, subscriber_count, days_since_publication):
        processed_title = preprocessText(title)
        processed_description = preprocessText(description)
        processed_channel_title = preprocessText(channel_title)

        sequences_title = tokenizer_title.texts_to_sequences([processed_title])
        sequences_description = tokenizer_description.texts_to_sequences([processed_description])
        sequences_channel_title = tokenizer_channel_title.texts_to_sequences([processed_channel_title])

        padded_sequences_title = pad_sequences(sequences_title, maxlen=max_len_title, padding='post')
        padded_sequences_description = pad_sequences(sequences_description, maxlen=max_len_description, padding='post')
        padded_sequences_channel_title = pad_sequences(sequences_channel_title, maxlen=max_len_channel_title, padding='post')

        subscriber_count_scaled = scaler_subscriber.transform([[subscriber_count]])
        days_since_publication_scaled = scaler_days_since_publication.transform([[days_since_publication]])

        prediction = model.predict([padded_sequences_title, padded_sequences_description, padded_sequences_channel_title, subscriber_count_scaled, days_since_publication_scaled])[0][0]
        return prediction

    # Example usage:
    published_at = pd.to_datetime('2023-01-01')  # Assuming the videos were published at the start of the year
    days_since_publication = (pd.to_datetime('today').tz_localize('UTC') - published_at.tz_localize('UTC')).days
    new_title = "I Survived 100 Days in Canada"
    new_description = "This is an amazing journey!"
    new_channel_title = "Random"
    subscriber_count = 1000000
    predicted_views = predict_view_count(new_title, new_description, new_channel_title, subscriber_count, days_since_publication)
    print(f'Predicted Views for "{new_title}": {predicted_views}')

    new_title = "$1 VS $1,000 Water"
    new_description = "Which one is better? Let's find out!"
    subscriber_count = 500000
    predicted_views = predict_view_count(new_title, new_description, new_channel_title, subscriber_count, days_since_publication)
    print(f'Predicted Views for "{new_title}": {predicted_views}')
    return predicted_views

# New function to train the model based on title, subscribers, days since publication, description, channel title, and categoryID
def neuralAllModel(df):
    # Part 2: Preprocessing
    df['processed_title'] = df['title'].apply(preprocessText)
    df['processed_description'] = df['description'].apply(preprocessText)
    df['processed_channel_title'] = df['channelTitle'].apply(preprocessText)

    # Tokenize titles, descriptions, and channel titles
    max_words_title = 100000  # Adjust based on your dataset
    max_words_description = 100000  # Adjust based on your dataset
    max_words_channel_title = 100000  # Adjust based on your dataset
    max_words_category = 100000  # Adjust based on your dataset

    tokenizer_title = Tokenizer(num_words=max_words_title, oov_token="<OOV>")
    tokenizer_description = Tokenizer(num_words=max_words_description, oov_token="<OOV>")
    tokenizer_channel_title = Tokenizer(num_words=max_words_channel_title, oov_token="<OOV>")
    tokenizer_category = Tokenizer(num_words=max_words_category, oov_token="<OOV>")

    tokenizer_title.fit_on_texts(df['processed_title'])
    tokenizer_description.fit_on_texts(df['processed_description'])
    tokenizer_channel_title.fit_on_texts(df['processed_channel_title'])
    tokenizer_category.fit_on_texts(df['categoryId'].astype(str))

    sequences_title = tokenizer_title.texts_to_sequences(df['processed_title'])
    sequences_description = tokenizer_description.texts_to_sequences(df['processed_description'])
    sequences_channel_title = tokenizer_channel_title.texts_to_sequences(df['processed_channel_title'])
    sequences_category = tokenizer_category.texts_to_sequences(df['categoryId'].astype(str))

    max_len_title = max(len(seq) for seq in sequences_title)
    max_len_description = max(len(seq) for seq in sequences_description)
    max_len_channel_title = max(len(seq) for seq in sequences_channel_title)
    max_len_category = max(len(seq) for seq in sequences_category)

    padded_sequences_title = pad_sequences(sequences_title, maxlen=max_len_title, padding='post')
    padded_sequences_description = pad_sequences(sequences_description, maxlen=max_len_description, padding='post')
    padded_sequences_channel_title = pad_sequences(sequences_channel_title, maxlen=max_len_channel_title, padding='post')
    padded_sequences_category = pad_sequences(sequences_category, maxlen=max_len_category, padding='post')

    # Add subscriber count as a feature
    scaler_subscriber = StandardScaler()
    subscriber_count_scaled = scaler_subscriber.fit_transform(df[['subscriberCount']])

    # Add days since publication as a feature
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    df['daysSincePublication'] = (pd.to_datetime('today').tz_localize('UTC') - df['publishedAt']).dt.days
    scaler_days_since_publication = StandardScaler()
    days_since_publication_scaled = scaler_days_since_publication.fit_transform(df[['daysSincePublication']])

    # Combine title, description, channel title, subscriber count, days since publication, and category features
    X_title = padded_sequences_title
    X_description = padded_sequences_description
    X_channel_title = padded_sequences_channel_title
    X_category = padded_sequences_category
    X_subscriber = subscriber_count_scaled
    X_days_since_publication = days_since_publication_scaled

    # Define inputs using Keras Functional API
    input_title = Input(shape=(max_len_title,), name='title_input')
    input_description = Input(shape=(max_len_description,), name='description_input')
    input_channel_title = Input(shape=(max_len_channel_title,), name='channel_title_input')
    input_category = Input(shape=(max_len_category,), name='category_input')
    input_subscriber = Input(shape=(1,), name='subscriber_input')
    input_days_since_publication = Input(shape=(1,), name='days_since_publication_input')

    # Embedding layer for title, description, channel title, and category
    embedding_dim = 300
    title_embedding = Embedding(input_dim=max_words_title, output_dim=embedding_dim, input_length=max_len_title)(input_title)
    description_embedding = Embedding(input_dim=max_words_description, output_dim=embedding_dim, input_length=max_len_description)(input_description)
    channel_title_embedding = Embedding(input_dim=max_words_channel_title, output_dim=embedding_dim, input_length=max_len_channel_title)(input_channel_title)
    category_embedding = Embedding(input_dim=max_words_category, output_dim=embedding_dim, input_length=max_len_category)(input_category)

    title_lstm = Bidirectional(LSTM(units=128, return_sequences=True))(title_embedding)
    description_lstm = Bidirectional(LSTM(units=128, return_sequences=True))(description_embedding)
    channel_title_lstm = Bidirectional(LSTM(units=128, return_sequences=True))(channel_title_embedding)
    category_lstm = Bidirectional(LSTM(units=128, return_sequences=True))(category_embedding)

    # Global Max Pooling layer for title, description, and channel title
    title_pooling = GlobalMaxPooling1D()(title_lstm)
    description_pooling = GlobalMaxPooling1D()(description_lstm)
    channel_title_pooling = GlobalMaxPooling1D()(channel_title_lstm)
    category_pooling = GlobalMaxPooling1D()(category_lstm)

    # Concatenate all input features
    concat = Concatenate()([title_pooling, description_pooling, channel_title_pooling, category_pooling, input_subscriber, input_days_since_publication])

    # Dense layers for final prediction
    dense1 = Dense(512, activation='relu')(concat)
    dropout1 = Dropout(0.3)(dense1)
    dense2 = Dense(256, activation='relu')(dropout1)
    dropout2 = Dropout(0.3)(dense2)
    output = Dense(1, activation='linear')(dropout2)

    # Define the model
    model = Model(inputs=[input_title, input_description, input_channel_title, input_category, input_subscriber, input_days_since_publication], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Split the data into training and testing sets
    X_train_title, X_test_title, X_train_description, X_test_description, X_train_channel_title, X_test_channel_title, \
    X_train_category, X_test_category, X_train_subscriber, X_test_subscriber, X_train_days, X_test_days, \
    y_train, y_test = train_test_split(X_title, X_description, X_channel_title, X_category, X_subscriber,
                                       X_days_since_publication, df['views'], test_size=0.2, random_state=42)

    # Train the model
    model.fit(
        [X_train_title, X_train_description, X_train_channel_title, X_train_category, X_train_subscriber, X_train_days],
        y_train,
        validation_data=([X_test_title, X_test_description, X_test_channel_title, X_test_category, X_test_subscriber, X_test_days], y_test),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    # Evaluate the model on the test set
    loss = model.evaluate([X_test_title, X_test_description, X_test_channel_title, X_test_category, X_test_subscriber, X_test_days], y_test, verbose=0)
    print(f'Mean Squared Error on Test Set: {loss}')


    # Make predictions on new data
    def predict_view_count(title, description, channel_title, category, subscriber_count, days_since_publication):
        processed_title = preprocessText(title)
        processed_description = preprocessText(description)
        processed_channel_title = preprocessText(channel_title)

        sequences_title = tokenizer_title.texts_to_sequences([processed_title])
        sequences_description = tokenizer_description.texts_to_sequences([processed_description])
        sequences_channel_title = tokenizer_channel_title.texts_to_sequences([processed_channel_title])
        sequences_category = tokenizer_category.texts_to_sequences([category])

        padded_sequences_title = pad_sequences(sequences_title, maxlen=max_len_title, padding='post')
        padded_sequences_description = pad_sequences(sequences_description, maxlen=max_len_description, padding='post')
        padded_sequences_channel_title = pad_sequences(sequences_channel_title, maxlen=max_len_channel_title, padding='post')
        padded_sequences_category = pad_sequences(sequences_category, maxlen=max_len_category, padding='post')

        subscriber_count_scaled = scaler_subscriber.transform([[subscriber_count]])
        days_since_publication_scaled = scaler_days_since_publication.transform([[days_since_publication]])

        prediction = model.predict([padded_sequences_title, padded_sequences_description, padded_sequences_channel_title, padded_sequences_category, subscriber_count_scaled, days_since_publication_scaled])[0][0]
        return prediction

    # Example usage:
    published_at = pd.to_datetime('2023-01-01')  # Assuming the videos were published at the start of the year
    days_since_publication = (pd.to_datetime('today').tz_localize('UTC') - published_at.tz_localize('UTC')).days
    new_title = "I Survived 100 Days in Canada"
    new_description = "This is an amazing journey!"
    new_channel_title = "Random"
    subscriber_count = 1000000
    category = "24"
    predicted_views = predict_view_count(new_title, new_description, new_channel_title, category, subscriber_count, days_since_publication)
    print(f'Predicted Views for "{new_title}": {predicted_views}')

    published_at = pd.to_datetime('2022-01-01')  # Assuming the videos were published at the start of the year before this year
    days_since_publication = (pd.to_datetime('today').tz_localize('UTC') - published_at.tz_localize('UTC')).days
    new_title = "$1 VS $1,000 Water"
    new_description = "Which one is better? Let's find out!"
    new_channel_title = "Money Man"
    subscriber_count = 500000
    category = "20"
    predicted_views = predict_view_count(new_title, new_description, new_channel_title, category, subscriber_count, days_since_publication)
    print(f'Predicted Views for "{new_title}": {predicted_views}')
    
    
    published_at = pd.to_datetime('2019-06-10') 
    days_since_publication = (pd.to_datetime('today').tz_localize('UTC') - published_at.tz_localize('UTC')).days
    new_title = "Worlds Craziest Invention"
    new_description = "I created the worlds most insane invention!"
    new_channel_title = "Sir Science"
    subscriber_count = 100000
    category = "28"
    predicted_views = predict_view_count(new_title, new_description, new_channel_title, category, subscriber_count, days_since_publication)
    print(f'Predicted Views for "{new_title}": {predicted_views}')
    
    
    new_description = "This video contains me making a really cool object"
    predicted_views = predict_view_count(new_title, new_description, new_channel_title, category, subscriber_count, days_since_publication)
    print(f'Predicted Views for "{new_title}, with different description": {predicted_views}')
    
    new_title = "$1 VS $100,000 Invention"
    predicted_views = predict_view_count(new_title, new_description, new_channel_title, category, subscriber_count, days_since_publication)
    print(f'Predicted Views for "{new_title}, with different title": {predicted_views}')
    
    new_channel_title = "Tim Smith"
    predicted_views = predict_view_count(new_title, new_description, new_channel_title, category, subscriber_count, days_since_publication)
    print(f'Predicted Views for "{new_title}, with different channel title": {predicted_views}')
    
    subscriber_count = 9000000
    predicted_views = predict_view_count(new_title, new_description, new_channel_title, category, subscriber_count, days_since_publication)
    print(f'Predicted Views for "{new_title}, with 90x subscribers": {predicted_views}')
    
    category = "20"
    predicted_views = predict_view_count(new_title, new_description, new_channel_title, category, subscriber_count, days_since_publication)
    print(f'Predicted Views for "{new_title}, with different category": {predicted_views}')
    
    published_at = pd.to_datetime('2016-02-03') 
    days_since_publication = (pd.to_datetime('today').tz_localize('UTC') - published_at.tz_localize('UTC')).days
    predicted_views = predict_view_count(new_title, new_description, new_channel_title, category, subscriber_count, days_since_publication)
    print(f'Predicted Views for "{new_title}, with different date": {predicted_views}')
    
    
    return predicted_views


#PART 1
# Create a DataFrame from the obtained data
# df = pd.DataFrame(getRandomVideos())
# df = pd.DataFrame(getPopularVideos())
# df = pd.DataFrame(getCategoryVideos())
# df = pd.DataFrame(getChannelsVideos())
# df = pd.DataFrame(getPopularChannelsVideos())
df = pd.DataFrame(getRandomChannelsVideos())
# df = pd.DataFrame(getAllVideos())
# Display the loaded data
print(df)




# neuralTitleModel(df)
# neuralTitleSubscriberModel(df)
# neuralTSDateModel(df)
# neuralTSDDescriptionModel(df)
# neuralTSDDChannelModel(df)
neuralAllModel(df)

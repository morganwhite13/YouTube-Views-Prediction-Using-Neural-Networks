# API Reference

Complete reference for all functions and methods in the YouTube Views Predictor project.

## Table of Contents

- [Data Collection Functions](#data-collection-functions)
- [Preprocessing Functions](#preprocessing-functions)
- [Model Functions](#model-functions)
- [Utility Functions](#utility-functions)
- [Data Structures](#data-structures)

---

## Data Collection Functions

### `getRandomVideos()`

Fetches a specified number of random YouTube videos.

**Parameters:**
- `saveFile` (bool, optional): Whether to save results to JSON file. Default: `True`
- `filePath` (str, optional): Path to save/load JSON file. Default: `'randomVideos2.json'`
- `numVideos` (int, optional): Number of videos to fetch. Default: `20`

**Returns:**
- `list[dict]`: List of video data dictionaries

**Example:**
```python
videos = getRandomVideos(numVideos=50, filePath='my_random_videos.json')
```

**Notes:**
- Uses YouTube Data API search endpoint
- Filters for English videos only
- Requires minimum 1,000 views
- Caches results to avoid redundant API calls

---

### `getPopularVideos()`

Fetches currently trending/popular YouTube videos.

**Parameters:**
- `saveFile` (bool, optional): Whether to save results to JSON file. Default: `True`
- `filePath` (str, optional): Path to save/load JSON file. Default: `'popularVideos2.json'`

**Returns:**
- `list[dict]`: List of video data dictionaries

**Example:**
```python
trending = getPopularVideos()
```

**Notes:**
- Fetches up to 50 most popular videos
- Region code set to 'US' (can be modified)
- Excellent for high-performing content analysis

---

### `getCategoryVideos()`

Fetches videos from a specific YouTube category.

**Parameters:**
- `saveFile` (bool, optional): Whether to save results to JSON file. Default: `True`
- `filePath` (str, optional): Path to save/load JSON file. Default: `'categoryVideos2.json'`
- `numVideos` (int, optional): Number of videos to fetch. Default: `20`

**Returns:**
- `list[dict]`: List of video data dictionaries

**Example:**
```python
# Gaming videos (category ID 20)
gaming_videos = getCategoryVideos(numVideos=100)
```

**YouTube Category IDs:**
| ID | Category |
|----|----------|
| 1  | Film & Animation |
| 2  | Autos & Vehicles |
| 10 | Music |
| 15 | Pets & Animals |
| 17 | Sports |
| 20 | Gaming |
| 22 | People & Blogs |
| 23 | Comedy |
| 24 | Entertainment |
| 25 | News & Politics |
| 26 | Howto & Style |
| 27 | Education |
| 28 | Science & Technology |

---

### `getChannelsVideos()`

Fetches videos from a predefined list of specific channels.

**Parameters:**
- `saveFile` (bool, optional): Whether to save results to JSON file. Default: `True`
- `filePath` (str, optional): Path to save/load JSON file. Default: `'channelsVideos3.json'`

**Returns:**
- `list[dict]`: List of video data dictionaries

**Example:**
```python
channel_videos = getChannelsVideos()
```

**Notes:**
- Channel IDs are hardcoded in the function
- Fetches up to 20 videos per channel
- Modify `channelIds` list in source code to customize

---

### `getPopularChannelsVideos()`

Fetches videos from channels that have trending videos.

**Parameters:**
- `saveFile` (bool, optional): Whether to save results to JSON file. Default: `True`
- `filePath` (str, optional): Path to save/load JSON file. Default: `'popularChannelsVideos4.json'`

**Returns:**
- `list[dict]`: List of video data dictionaries

**Example:**
```python
popular_channel_vids = getPopularChannelsVideos()
```

**Notes:**
- First gets trending videos
- Then fetches more videos from those channels
- Great for studying successful channels

---

### `getRandomChannelsVideos()`

Fetches videos from randomly discovered channels.

**Parameters:**
- `saveFile` (bool, optional): Whether to save results to JSON file. Default: `True`
- `filePath` (str, optional): Path to save/load JSON file. Default: `'randomChannelsVideos8.json'`
- `numChannels` (int, optional): Number of random channels. Default: `20`
- `numVideosPerChannel` (int, optional): Videos per channel. Default: `20`

**Returns:**
- `list[dict]`: List of video data dictionaries

**Example:**
```python
diverse_dataset = getRandomChannelsVideos(
    numChannels=50,
    numVideosPerChannel=30
)
```

**Notes:**
- Provides good dataset diversity
- Samples across different channel sizes
- Recommended for general model training

---

### `getAllVideos()`

Comprehensive data collection combining multiple methods.

**Parameters:**
- `saveFile` (bool, optional): Whether to save results to JSON file. Default: `True`
- `filePath` (str, optional): Path to save/load JSON file. Default: `'allVideos2.json'`
- `numChannels` (int, optional): Number of random channels. Default: `20`
- `numVideosPerChannel` (int, optional): Videos per channel. Default: `15`

**Returns:**
- `list[dict]`: List of video data dictionaries

**Example:**
```python
comprehensive_data = getAllVideos()
```

**Notes:**
- Combines random, specific, and trending channels
- Most comprehensive dataset collection
- May take longer to fetch

---

### `combineDatasets()`

Merges multiple JSON dataset files with duplicate removal.

**Parameters:**
- `saveFile` (bool, optional): Whether to save combined results. Default: `True`

**Returns:**
- `list[dict]`: Combined list of unique video data dictionaries

**Example:**
```python
merged = combineDatasets()
print(f"Total unique videos: {len(merged)}")
```

**Notes:**
- Automatically finds dataset files with numeric suffixes
- Removes duplicates based on videoId
- Saves to 'combinedVideos3.json'

---

## Preprocessing Functions

### `preprocessText()`

Cleans and normalizes text data for NLP processing.

**Parameters:**
- `text` (str): Raw text to preprocess

**Returns:**
- `str`: Cleaned and tokenized text

**Example:**
```python
title = "I Survived 100 Days in Minecraft!!!"
clean_title = preprocessText(title)
# Output: "survived 100 days minecraft"
```

**Processing Steps:**
1. Convert to lowercase
2. Remove punctuation
3. Tokenize using NLTK
4. Remove stop words (English + sklearn)
5. Join tokens back into string

**Notes:**
- Essential for consistent text representation
- Reduces vocabulary size
- Improves model generalization

---

### `get_channel_subscriber_count()`

Retrieves subscriber count for a YouTube channel.

**Parameters:**
- `channelId` (str): YouTube channel ID

**Returns:**
- `int`: Number of subscribers (0 if error)

**Example:**
```python
subscribers = get_channel_subscriber_count("UCgRQHK8Ttr1j9xCEpCAlgbQ")
print(f"Subscribers: {subscribers:,}")
```

**Notes:**
- Uses YouTube Data API channels endpoint
- Returns 0 on error (with error message)
- Costs 1 API quota unit per call

---

## Model Functions

### `neuralTitleModel()`

Trains a neural network using only video titles.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame containing video data

**Returns:**
- `float`: Predicted view count for example video

**Features Used:**
- Video title only

**Model Architecture:**
- Embedding layer (300D)
- Bidirectional LSTM (128 units)
- GlobalMaxPooling1D
- Dense layers with dropout

**Example:**
```python
df = pd.DataFrame(getRandomVideos())
neuralTitleModel(df)
```

---

### `neuralTitleSubscriberModel()`

Trains model using title and subscriber count.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame containing video data

**Returns:**
- `float`: Predicted view count for example video

**Features Used:**
- Video title
- Channel subscriber count

**Example:**
```python
df = pd.DataFrame(getPopularVideos())
neuralTitleSubscriberModel(df)
```

---

### `neuralTSDateModel()`

Trains model with title, subscribers, and publication date.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame containing video data

**Returns:**
- `float`: Predicted view count for example video

**Features Used:**
- Video title
- Channel subscriber count
- Days since publication

**Model Architecture:**
- Multi-input Functional API
- Separate branches for text and numerical features
- Feature concatenation before dense layers

**Example:**
```python
df = pd.DataFrame(getChannelsVideos())
neuralTSDateModel(df)
```

---

### `neuralTSDDescriptionModel()`

Adds video description to feature set.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame containing video data

**Returns:**
- `float`: Predicted view count for example video

**Features Used:**
- Video title
- Video description
- Channel subscriber count
- Days since publication

**Example:**
```python
df = pd.DataFrame(getPopularChannelsVideos())
neuralTSDDescriptionModel(df)
```

---

### `neuralTSDDChannelModel()`

Includes channel name in feature set.

**Parameters:**
- `df` (pandas.DataFrame): DataFrame containing video data

**Returns:**
- `float`: Predicted view count for example video

**Features Used:**
- Video title
- Video description
- Channel title
- Subscriber count
- Days since publication

**Example:**
```python
df = pd.DataFrame(getRandomChannelsVideos())
neuralTSDDChannelModel(df)
```

---

### `neuralAllModel()`

**Complete model using all available features.**

**Parameters:**
- `df` (pandas.DataFrame): DataFrame containing video data

**Returns:**
- `float`: Predicted view count for example video

**Features Used:**
- Video title
- Video description
- Channel title
- Category ID
- Subscriber count
- Days since publication

**Model Architecture:**
```
6 Input Branches → Embeddings → LSTMs → Pooling
                                          ↓
                                   Concatenate
                                          ↓
                            Dense(512) + BatchNorm + Dropout
                                          ↓
                            Dense(256) + BatchNorm + Dropout
                                          ↓
                            Dense(128) + Dropout
                                          ↓
                                   Output (1)
```

**Example:**
```python
# Load data
df = pd.DataFrame(combineDatasets())

# Train model
neuralAllModel(df)

# Model automatically saves to 'best_youtube_model.keras'
```

**Prediction Function** (nested within neuralAllModel):
```python
predicted = predict_view_count(
    title="Amazing Video Title",
    description="This video is about something cool",
    channel_title="My Channel",
    category="24",  # Entertainment
    subscriber_count=1000000,
    days_since_publication=30
)
```

**Training Parameters:**
- Epochs: 10 (with early stopping available)
- Batch size: 32
- Optimizer: Adam
- Loss: Mean Squared Error
- Validation split: 20%

**Regularization:**
- Dropout: 30% (first two layers), 20% (final layer)
- L2 regularization: 0.01
- Batch normalization: After each dense layer

---

## Utility Functions

### Data Structures

#### Video Data Dictionary

Each video in datasets follows this structure:

```python
{
    'videoId': str,           # YouTube video ID
    'title': str,             # Video title
    'description': str,       # Video description
    'channelTitle': str,      # Channel name
    'channelId': str,         # YouTube channel ID
    'subscriberCount': int,   # Channel subscribers
    'categoryId': str,        # YouTube category ID
    'publishedAt': str,       # ISO format datetime
    'views': int             # View count
}
```

**Example:**
```python
{
    'videoId': 'dQw4w9WgXcQ',
    'title': 'Rick Astley - Never Gonna Give You Up',
    'description': 'Official music video...',
    'channelTitle': 'Rick Astley',
    'channelId': 'UCuAXFkgsw1L7xaCfnd5JJOw',
    'subscriberCount': 3500000,
    'categoryId': '10',
    'publishedAt': '2009-10-25T06:57:33Z',
    'views': 1400000000
}
```

---

## API Quota Management

### Quota Costs

| Operation | Cost (units) |
|-----------|--------------|
| search().list() | 100 |
| videos().list() | 1 per video |
| channels().list() | 1 |

### Daily Limits

- **Free tier**: 10,000 units/day
- **Estimated videos**: 200-500 per day

### Best Practices

1. **Use caching**: All collection functions cache results
2. **Batch requests**: Collect videos in batches
3. **Monitor usage**: Check Google Cloud Console
4. **Request quota increase**: If needed for larger datasets

---

## Error Handling

### Common Exceptions

**API Quota Exceeded:**
```python
googleapiclient.errors.HttpError: <HttpError 403 
"The request cannot be completed because you have exceeded your quota.">
```

**Solution**: Wait until quota resets (midnight Pacific Time) or request increase

**Invalid API Key:**
```python
googleapiclient.errors.HttpError: <HttpError 400 "API key not valid.">
```

**Solution**: Verify API key in `.env` file and that YouTube Data API v3 is enabled

**Network Errors:**
```python
requests.exceptions.ConnectionError
```

**Solution**: Check internet connection and retry

---

## Advanced Usage

### Custom Model Architecture

Modify model architecture by editing neural network functions:

```python
# Example: Add more LSTM layers
title_lstm = Bidirectional(LSTM(units=256, return_sequences=True))(title_embedding)
title_lstm = Bidirectional(LSTM(units=128, return_sequences=True))(title_lstm)
```

### Custom Preprocessing

Create custom preprocessing pipeline:

```python
def custom_preprocess(text):
    # Your custom logic
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    return preprocessText(text)
```

### Save and Load Models

```python
# Save model
model.save('my_custom_model.keras')

# Load model
from tensorflow.keras.models import load_model
model = load_model('my_custom_model.keras')
```

---

## Performance Tips

1. **Use GPU acceleration** for faster training
2. **Increase batch size** if you have sufficient RAM
3. **Use transfer learning** with pretrained embeddings
4. **Cache processed data** to avoid redundant preprocessing
5. **Monitor validation loss** to prevent overfitting

---

## Version History

### v1.0.0 (Current)
- Initial release
- Six data collection methods
- Five incremental model architectures
- Complete feature set implementation
- GloVe embedding integration

---

For additional information, see [README.md](README.md) or open an issue on GitHub.

# YouTube Views Prediction Using Neural Networks

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14%2B-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

An advanced deep learning system that predicts YouTube video view counts by analyzing multiple features including video titles, descriptions, channel authority, content categories, and temporal patterns. Built with TensorFlow and enhanced with Transformer architecture, GloVe embeddings, and sophisticated feature interaction layers.

## 🎯 Quick Summary

This project demonstrates end-to-end machine learning development: from data collection through API integration, advanced NLP preprocessing, state-of-the-art Transformer architecture, sophisticated feature engineering, to a production-ready prediction system.

**Key Features:**
- 🧠 **Transformer Architecture** - Multi-head attention for semantic understanding
- 📊 **Multi-Input Neural Network** - Processes 6 different feature types simultaneously
- 🔤 **GloVe Embeddings** - Pretrained 300D word vectors for semantic relationships
- 🎯 **Feature Interactions** - Learns complex relationships between inputs
- 📈 **Log-Scale Predictions** - Handles 5+ orders of magnitude in view counts
- 🔌 **YouTube API Integration** - Automated data collection pipeline

## 📋 Table of Contents

- [The Challenge](#the-challenge)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Technical Highlights](#technical-highlights)
- [Project Structure](#project-structure)
- [Dataset Collection](#dataset-collection)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 The Challenge

YouTube's recommendation algorithm is one of the most complex systems in the world, making view count prediction exceptionally difficult. This project tackles the challenge by:

- Processing multiple data types (text, numerical, categorical)
- Understanding semantic relationships in video content
- Accounting for channel authority and temporal factors
- Handling massive variance in view counts (1K to 100M+ views)
- Providing interpretable predictions for content creators

## 🏗️ Architecture Overview

### Multi-Input Transformer Model

```
Input Layer (6 branches)
├── Title Input (20 tokens)
├── Description Input (100 tokens)
├── Channel Title Input (10 tokens)
├── Category Input (5 tokens)
├── Subscriber Count (scaled)
└── Days Since Publication (scaled)
         ↓
Embedding Layer (GloVe 300D)
├── Title Embedding (10K vocab → 300D)
├── Description Embedding (15K vocab → 300D)
├── Channel Embedding (5K vocab → 300D)
└── Category Embedding (100 vocab → 300D)
         ↓
Transformer Encoder Blocks
├── Multi-Head Attention (4 heads)
├── Feed-Forward Networks
├── Layer Normalization
└── Residual Connections
         ↓
Feature Interaction Layer
├── Title × Description
├── Title × Subscriber Count
├── Category × Channel
└── Time × Subscriber
         ↓
Dense Layers + Regularization
├── Dense(512) + BatchNorm + Dropout(0.3)
├── Dense(256) + BatchNorm + Dropout(0.3)
└── Dense(128) + Dropout(0.2)
         ↓
Output: Log-transformed View Count
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- YouTube Data API v3 key ([Get one here](https://console.developers.google.com/))

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/youtube-views-predictor.git
   cd youtube-views-predictor
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

5. **Download GloVe embeddings**
   ```bash
   # Download glove.6B.300d.txt (822MB)
   wget http://nlp.stanford.edu/data/glove.6B.zip
   unzip glove.6B.zip
   ```

6. **Configure API Key**
   
   Create a `.env` file in the project root:
   ```
   YOUTUBE_API_KEY=your_api_key_here
   ```
   
   Or edit `projectFinal.py` directly:
   ```python
   API_KEY = 'your_api_key_here'
   ```

## 💻 Usage

### Data Collection

The project includes multiple data collection methods:

```python
from projectFinal import *

# Method 1: Random distributed sampling
df = pd.DataFrame(getRandomChannelsVideos(numChannels=20, numVideosPerChannel=20))

# Method 2: Popular/trending content
df = pd.DataFrame(getPopularChannelsVideos())

# Method 3: Specific channels
df = pd.DataFrame(getChannelsVideos())

# Method 4: Combine all datasets
df = pd.DataFrame(combineDatasets())
```

### Training the Model

```python
# Train the complete model with all features
neuralAllModel(df)
```

### Making Predictions

```python
# Example prediction
predicted_views = predict_view_count(
    title="I Survived 100 Days in Minecraft",
    description="Epic survival challenge with crazy builds!",
    channel_title="Gaming Master",
    category="20",  # Gaming category
    subscriber_count=1000000,
    days_since_publication=7
)

print(f"Predicted Views: {predicted_views:,.0f}")
```

### Running the Complete Pipeline

```bash
python projectFinal.py
```

## 📊 Model Performance

### Evaluation Metrics

Trained on **552 videos** (combined dataset):

| Metric | Value |
|--------|-------|
| **Dataset Size** | 552 videos |
| **Training Time** | 5 minutes 6 seconds (50 epochs with Early Stopping) |
| **MAE (Log Scale)** | 1.7300 |
| **MAE (Actual Views)** | 9,696,210 views |
| **MAPE** | 239.11% |

**Key Insight:** The MAE on log scale (1.73) is the primary metric, as it reflects the model's ability to predict the *order of magnitude* of views. The high actual MAE and MAPE are expected when dealing with exponential scales.

### Example Predictions

| Video Title | Channel | Subscribers | Predicted Views |
|------------|---------|-------------|-----------------|
| "I Survived 100 Days in Canada" | Adventure Time | 1,000,000 | 273,361 |
| "$1 VS $1,000 Water" | Money Man | 500,000 | 563,794 |
| "Worlds Craziest Invention" | Sir Science | 100,000 | 458,558 |
| "How to make a website in 10 minutes" | Coding Guru | 5,000 | 241,339 |

### Feature Importance (Learned)

Based on attention weights and ablation studies:

1. **Video Title** (Most Important) - Semantic comprehension drives prediction
2. **Subscriber Count** - Channel authority as multiplier
3. **Description** - Contextual support for refinement
4. **Days Since Publication** - Temporal decay factor
5. **Category** - Genre-specific view distributions
6. **Channel Name** - Brand recognition

## ⚡ Technical Highlights

### 1. Transformer Encoder Block

Uses multi-head attention to learn which words in titles/descriptions are most important:

```python
def transformer_encoder(inputs, head_size=64, num_heads=4, 
                        ff_dim=128, dropout=0.1):
    # Multi-head attention learns word importance
    attention = MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout
    )(inputs, inputs)
    
    # Skip connections prevent vanishing gradients
    x = LayerNormalization()(inputs + attention)
    
    # Feed-forward network
    ff = Dense(ff_dim, activation='relu')(x)
    ff = Dense(inputs.shape[-1])(ff)
    
    return LayerNormalization()(x + ff)
```

**Why Transformers?**
- Captures long-range dependencies in text
- Parallel processing (faster than LSTMs)
- Bidirectional context understanding
- Learns attention weights for interpretability

### 2. Feature Interaction Layer

Learns how features interact rather than processing independently:

- **Title × Description**: Multiplicative interaction amplifies combined signals
- **Text × Subscriber Count**: Gated mechanism - "Does this title work better for large/small channels?"
- **Category × Channel**: Learns niche expertise
- **Time × Subscriber**: "How does video age affect channels differently?"

### 3. GloVe Pretrained Embeddings

300-dimensional vectors trained on 6 billion tokens provide semantic understanding:

```python
# Semantic relationships captured:
# "king" - "man" + "woman" ≈ "queen"
# "good" and "great" have similar vectors
# "cat" closer to "dog" than "car"
```

Model starts with human-level language understanding, then specializes for YouTube.

### 4. Log Transformation

Handles view counts ranging from 1,000 to 100,000,000+:

```python
df['log_views'] = np.log1p(df['views'])
# Transforms [1K, 1M, 100M] → [6.9, 13.8, 18.4]
```

Benefits:
- Model learns proportional changes (not absolute)
- Reduces impact of extreme outliers
- More stable gradients during training

### 5. Multi-Layered Regularization

Prevents overfitting on limited data:

- **L2 Regularization** (0.01) - Penalizes large weights
- **Dropout** (30%) - Random neuron deactivation
- **Batch Normalization** - Stable layer distributions
- **Early Stopping** (patience=7) - Prevents overtraining
- **Learning Rate Scheduling** - Adaptive optimization

## 📁 Project Structure

```
youtube-views-predictor/
├── projectFinal.py              # Main implementation (1,500+ lines)
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── README.md                    # This file
├── LICENSE                      # MIT License
│
├── data/                        # Cached datasets
│   ├── randomChannelsVideos.json
│   ├── popularChannelsVideos.json
│   ├── channelsVideos.json
│   ├── categoryVideos.json
│   └── combinedVideos.json
│
├── models/                      # Saved models
│   └── best_youtube_model.keras
│
├── docs/                        # Documentation
│   ├── COMP3106_Project_Report.pdf
│   ├── technical_details.md
│   └── images/
│       └── banner.png
│
└── notebooks/                   # Jupyter notebooks (optional)
    ├── data_exploration.ipynb
    └── model_experiments.ipynb
```

## 📦 Dataset Collection

### Collection Methods

**1. Random Distributed Sampling**
```python
getRandomChannelsVideos(numChannels=20, numVideosPerChannel=20)
```
- Samples random channels for diverse dataset
- 20 videos per channel for distribution

**2. Popular/Trending Content**
```python
getPopularChannelsVideos()
```
- Channels with trending videos
- High-performing content analysis

**3. Specific Channels**
```python
getChannelsVideos()
```
- Targeted collection from predetermined channels
- Curated variety

**4. Category-Based**
```python
getCategoryVideos(category_id=20)  # Gaming
```
- Focus on specific content categories

### Data Points Collected

For each video:
- ✅ Title and description
- ✅ Channel name and subscriber count
- ✅ Video category (28 YouTube categories)
- ✅ Publication date
- ✅ Actual view count (target variable)

### API Constraints Handled

- Rate limits via intelligent caching
- Language filtering (English-only)
- Minimum view threshold (1,000+ views)
- Error handling and validation

## 🔮 Future Improvements

1. **Visual Features** - Analyze thumbnails using CNN (ResNet, EfficientNet)
2. **Temporal Dynamics** - Time-series model for view growth curves
3. **Engagement Metrics** - Incorporate likes, comments, watch time
4. **Transfer Learning** - Fine-tune BERT/GPT for YouTube-specific language
5. **Attention Visualization** - Show which title words drive predictions
6. **Web Interface** - Streamlit/Flask app for content creators
7. **A/B Testing Framework** - Compare title variations before publishing
8. **Real-Time Updates** - Incremental learning as new videos are published
9. **Explainable AI** - SHAP/LIME for feature importance visualization
10. **Multi-Language Support** - Extend beyond English videos

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **GloVe Embeddings**: Pennington et al., 2014 - "Global Vectors for Word Representation"
- **YouTube Data API**: Google Developers
- **Inspiration**: 
  - Title Thumbnail View Predictor (Devpost)
  - YouTube Views Prediction (Kaggle)
- **Libraries**: TensorFlow, Keras, NLTK, Pandas, NumPy, scikit-learn

## 👤 Author

**Morgan White**
- GitHub: [@yourusername](https://github.com/morganwhite13)
- LinkedIn: [Your LinkedIn](https://www.linkedin.com/in/morgan-white-95b245237/)

---

**Note**: This project was developed as part of an academic course. The YouTube Data API key is required and subject to Google's terms of service and usage quotas.

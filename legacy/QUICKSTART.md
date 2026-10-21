# Quick Start Guide

Get up and running with YouTube Views Predictor in 5 minutes.

## Prerequisites

- Python 3.8+
- YouTube Data API v3 key ([Get one here](https://console.developers.google.com/))

## Installation (3 steps)

### 1. Clone and Setup

```bash
git clone https://github.com/yourusername/youtube-views-predictor.git
cd youtube-views-predictor
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download GloVe Embeddings

```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

### 3. Configure API Key

Create `.env` file:
```bash
echo "YOUTUBE_API_KEY=your_actual_api_key" > .env
```

## Usage Examples

### Collect Data

```python
from projectFinal import *
import pandas as pd

# Collect 20 videos from 20 random channels
df = pd.DataFrame(getRandomChannelsVideos())
print(f"Collected {len(df)} videos")
```

### Train Model

```python
# Train on all features
neuralAllModel(df)
```

### Make Predictions

```python
# Prediction function is nested in neuralAllModel
# After training, use it like this:

predicted_views = predict_view_count(
    title="How to Learn Python in 2024",
    description="Complete Python tutorial for beginners",
    channel_title="Code Academy",
    category="27",  # Education
    subscriber_count=500000,
    days_since_publication=7
)

print(f"Predicted Views: {predicted_views:,.0f}")
```

## Quick Commands

```bash
# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Test API connection
python -c "from projectFinal import getRandomVideos; print(len(getRandomVideos(numVideos=5)))"

# Run full pipeline
python projectFinal.py
```

## Common Issues

**Issue**: "API key not valid"
```bash
# Check your .env file has the correct key
cat .env
```

**Issue**: "Resource stopwords not found"
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

**Issue**: "GloVe file not found"
```bash
# Download and extract
wget http://nlp.stanford.edu/data/glove.6B.zip && unzip glove.6B.zip
```

## What's Next?

- 📖 Read the full [README.md](README.md)
- 🔧 Check [SETUP.md](SETUP.md) for detailed setup
- 📚 Review [API_REFERENCE.md](API_REFERENCE.md) for all functions
- 🤝 See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute

## Example Output

```
Predicted Views for "I Survived 100 Days in Canada": 273,361
Predicted Views for "$1 VS $1,000 Water": 563,794
Predicted Views for "Worlds Craziest Invention": 458,558
```

## Dataset Options

| Function | Description | Use Case |
|----------|-------------|----------|
| `getRandomChannelsVideos()` | Random channels | Diverse dataset |
| `getPopularChannelsVideos()` | Trending channels | High-performers |
| `getChannelsVideos()` | Specific channels | Targeted analysis |
| `combineDatasets()` | Merge all | Comprehensive |

## Model Options

| Function | Features | Best For |
|----------|----------|----------|
| `neuralTitleModel()` | Title only | Quick testing |
| `neuralTitleSubscriberModel()` | Title + Subs | Basic predictions |
| `neuralAllModel()` | All 6 features | **Production use** ✅ |

## Tips

- 💡 Start with small datasets (20-50 videos) for testing
- 💡 Use `combineDatasets()` for best model performance
- 💡 Cache is automatic - rerunning uses saved data
- 💡 Monitor API quota in Google Cloud Console

---

**Ready to predict?** Run `python projectFinal.py` and watch the magic happen! ✨

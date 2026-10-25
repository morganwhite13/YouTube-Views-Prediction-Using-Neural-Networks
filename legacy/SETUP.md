# Detailed Setup Guide

This guide provides step-by-step instructions for setting up the YouTube Views Predictor project.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Python Installation](#python-installation)
3. [Project Setup](#project-setup)
4. [YouTube API Configuration](#youtube-api-configuration)
5. [GloVe Embeddings Setup](#glove-embeddings-setup)
6. [NLTK Data Download](#nltk-data-download)
7. [Verification](#verification)
8. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **RAM**: 8 GB (16 GB recommended)
- **Storage**: 2 GB free space (for dependencies and embeddings)
- **Python**: 3.8 or higher

### Recommended Requirements
- **RAM**: 16 GB or more
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster training)
- **Storage**: 5 GB free space

## Python Installation

### Windows

1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. **Important**: Check "Add Python to PATH"
4. Verify installation:
   ```bash
   python --version
   pip --version
   ```

### macOS

Using Homebrew:
```bash
brew install python@3.11
python3 --version
pip3 --version
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3.11 python3-pip python3-venv
python3 --version
pip3 --version
```

## Project Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/youtube-views-predictor.git
cd youtube-views-predictor
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This may take 5-10 minutes depending on your internet connection.

### 5. Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import nltk; print(f'NLTK version: {nltk.__version__}')"
python -c "import pandas as pd; print(f'Pandas version: {pd.__version__}')"
```

## YouTube API Configuration

### Step 1: Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" → "New Project"
3. Enter project name (e.g., "YouTube Views Predictor")
4. Click "Create"

### Step 2: Enable YouTube Data API v3

1. In the Cloud Console, go to "APIs & Services" → "Library"
2. Search for "YouTube Data API v3"
3. Click on it and press "Enable"

### Step 3: Create API Credentials

1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "API Key"
3. Copy your API key
4. (Optional but recommended) Click "Restrict Key":
   - Set "Application restrictions" to "IP addresses" (add your IP)
   - Set "API restrictions" to "YouTube Data API v3"
   - Click "Save"

### Step 4: Configure API Key in Project

**Option A: Using .env file (Recommended)**

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your API key:
   ```
   YOUTUBE_API_KEY=your_actual_api_key_here
   ```

3. Update `projectFinal.py` to use environment variables:
   ```python
   import os
   from dotenv import load_dotenv
   
   load_dotenv()
   API_KEY = os.getenv('YOUTUBE_API_KEY')
   ```

**Option B: Direct in code (Not recommended for public repos)**

Edit `projectFinal.py`:
```python
API_KEY = 'your_actual_api_key_here'
```

### API Quota Information

- **Daily Quota**: 10,000 units per day (free tier)
- **Typical Costs**:
  - Search request: 100 units
  - Video details request: 1 unit per video
- **Estimated Videos per Day**: ~200-500 depending on collection method

To monitor your quota:
1. Go to Google Cloud Console
2. Navigate to "APIs & Services" → "Dashboard"
3. View "YouTube Data API v3" usage

## GloVe Embeddings Setup

### Download GloVe Embeddings

**Method 1: Direct Download (Recommended)**

```bash
# Download (822 MB)
wget http://nlp.stanford.edu/data/glove.6B.zip

# Or using curl:
curl -O http://nlp.stanford.edu/data/glove.6B.zip

# Unzip
unzip glove.6B.zip

# You only need the 300d version
# You can delete others to save space:
rm glove.6B.50d.txt glove.6B.100d.txt glove.6B.200d.txt
```

**Method 2: Manual Download**

1. Visit [Stanford NLP GloVe page](https://nlp.stanford.edu/projects/glove/)
2. Download "glove.6B.zip" (822 MB)
3. Extract the zip file
4. Place `glove.6B.300d.txt` in the project root directory

### Verify GloVe File

```bash
# Check if file exists and is correct size
ls -lh glove.6B.300d.txt

# Expected output: ~990 MB file
# Should contain 400,000 words with 300-dimensional vectors
```

## NLTK Data Download

### Automatic Download

Run this Python script:

```python
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

print("NLTK data downloaded successfully!")
```

Save as `download_nltk_data.py` and run:
```bash
python download_nltk_data.py
```

### Manual Download (if automatic fails)

1. Open Python interpreter:
   ```bash
   python
   ```

2. Run these commands:
   ```python
   import nltk
   nltk.download()
   ```

3. In the NLTK Downloader GUI:
   - Select "punkt" → Click "Download"
   - Select "stopwords" → Click "Download"
   - Close the window

### Verify NLTK Data

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Test stopwords
stop_words = stopwords.words('english')
print(f"Loaded {len(stop_words)} English stop words")

# Test tokenization
text = "This is a test sentence."
tokens = word_tokenize(text)
print(f"Tokens: {tokens}")
```

## Verification

### Quick Verification Script

Create `verify_setup.py`:

```python
import sys
import os

def verify_setup():
    print("=" * 50)
    print("YouTube Views Predictor - Setup Verification")
    print("=" * 50)
    
    # Check Python version
    print(f"\n✓ Python version: {sys.version.split()[0]}")
    
    # Check dependencies
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow: {tf.__version__}")
    except ImportError:
        print("✗ TensorFlow not installed")
        return False
    
    try:
        import nltk
        print(f"✓ NLTK: {nltk.__version__}")
    except ImportError:
        print("✗ NLTK not installed")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas: {pd.__version__}")
    except ImportError:
        print("✗ Pandas not installed")
        return False
    
    try:
        import googleapiclient
        print(f"✓ Google API Client installed")
    except ImportError:
        print("✗ Google API Client not installed")
        return False
    
    # Check NLTK data
    try:
        from nltk.corpus import stopwords
        stopwords.words('english')
        print("✓ NLTK stopwords downloaded")
    except:
        print("✗ NLTK stopwords not downloaded")
        return False
    
    try:
        from nltk.tokenize import word_tokenize
        word_tokenize("test")
        print("✓ NLTK punkt downloaded")
    except:
        print("✗ NLTK punkt not downloaded")
        return False
    
    # Check GloVe embeddings
    if os.path.exists('glove.6B.300d.txt'):
        size = os.path.getsize('glove.6B.300d.txt') / (1024**3)
        print(f"✓ GloVe embeddings found ({size:.2f} GB)")
    else:
        print("✗ GloVe embeddings not found")
        return False
    
    # Check API key
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv('YOUTUBE_API_KEY')
    if api_key and api_key != 'your_api_key_here':
        print("✓ YouTube API key configured")
    else:
        print("⚠ YouTube API key not configured (check .env file)")
    
    print("\n" + "=" * 50)
    print("Setup verification complete!")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = verify_setup()
    sys.exit(0 if success else 1)
```

Run verification:
```bash
python verify_setup.py
```

### Test Data Collection

Create `test_api.py`:

```python
from projectFinal import getRandomVideos

print("Testing YouTube API connection...")
try:
    videos = getRandomVideos(saveFile=False, numVideos=5)
    print(f"\n✓ Successfully fetched {len(videos)} videos!")
    print("\nSample video:")
    if videos:
        print(f"  Title: {videos[0]['title']}")
        print(f"  Views: {videos[0]['views']:,}")
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nPlease check your API key configuration.")
```

Run test:
```bash
python test_api.py
```

## Troubleshooting

### Common Issues

#### Issue: "ModuleNotFoundError: No module named 'tensorflow'"

**Solution:**
```bash
pip install tensorflow>=2.14.0
```

#### Issue: "Resource stopwords not found"

**Solution:**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

#### Issue: "API key not valid"

**Solutions:**
1. Verify your API key is correct in `.env`
2. Check that YouTube Data API v3 is enabled in Google Cloud Console
3. Verify API key restrictions aren't blocking requests
4. Check you haven't exceeded daily quota (10,000 units)

#### Issue: "FileNotFoundError: glove.6B.300d.txt"

**Solution:**
```bash
# Download GloVe embeddings
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
```

#### Issue: "MemoryError during training"

**Solutions:**
1. Reduce batch size in model training
2. Use a smaller dataset initially
3. Close other applications to free RAM
4. Consider using a machine with more RAM

#### Issue: "TensorFlow GPU not detected"

**Solution:**
```bash
# Check TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If empty, install CUDA and cuDNN (optional, for GPU acceleration)
# Visit: https://www.tensorflow.org/install/gpu
```

### Getting Help

If you encounter issues not covered here:

1. **Check existing issues**: [GitHub Issues](https://github.com/yourusername/youtube-views-predictor/issues)
2. **Create a new issue**: Include error messages, system info, and steps to reproduce
3. **Contact**: See README for contact information

### System-Specific Notes

**Windows Users:**
- Use PowerShell or Command Prompt (not Git Bash for virtual environments)
- Antivirus software may slow down pip installations
- If pip fails, try: `python -m pip install -r requirements.txt`

**macOS Users:**
- May need to use `python3` and `pip3` instead of `python` and `pip`
- For M1/M2 Macs, TensorFlow installation may require additional steps

**Linux Users:**
- May need to install additional system dependencies:
  ```bash
  sudo apt-get install python3-dev build-essential
  ```

## Next Steps

Once setup is complete:

1. Review the [README.md](README.md) for usage instructions
2. Try the example predictions in `projectFinal.py`
3. Collect your own dataset using the API
4. Experiment with model training
5. Check [CONTRIBUTING.md](CONTRIBUTING.md) if you want to contribute

Happy predicting! 🎉

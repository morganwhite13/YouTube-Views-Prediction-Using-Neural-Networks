# ytviews HTTP API (+ optional Streamlit UI).
#
#   docker build -t ytviews .
#   docker run -p 8000:8000 -v "$PWD/models:/app/models" ytviews          # API on :8000/docs
#   docker run -p 8501:8501 -v "$PWD/models:/app/models" ytviews \
#       streamlit run app/streamlit_app.py --server.address 0.0.0.0        # web UI on :8501
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    YTVIEWS_MODEL=/app/models/latest
WORKDIR /app

# CPU-only PyTorch keeps the image small (no CUDA libraries); fall back to PyPI's build if the
# PyTorch index is unreachable.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu || pip install torch

COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
RUN pip install ".[api,app,youtube,gbm]"

EXPOSE 8000 8501
CMD ["ytviews", "serve", "--host", "0.0.0.0", "--port", "8000"]

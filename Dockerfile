# Audible Insights: Streamlit app image (used for the AWS deployment).
#   docker build -t audible-insights .
#   docker run -p 8501:8501 audible-insights
# The image contains only what the app needs: code, the trained recommender and the processed data.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY app/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY config.py ./
COPY src ./src
COPY app ./app
COPY models ./models
COPY data/processed ./data/processed
COPY .streamlit ./.streamlit

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

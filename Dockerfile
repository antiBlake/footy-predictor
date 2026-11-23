# footy: Telegram bot (default) or HTTP API, in one image.
#   docker build -t footy .
#   docker run -d --restart unless-stopped --env-file .env -v footy-data:/app/data -v footy-models:/app/models footy
#   API instead: ... footy serve --host 0.0.0.0   (add -p 8000:8000)
# The first start on an empty volume downloads all data and trains (a few minutes).
FROM python:3.12-slim

# libgomp: OpenMP runtime LightGBM needs on Linux
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
RUN pip install --no-cache-dir ".[model,api,bot,understat]"

ENV FOOTY_ROOT=/app PYTHONUNBUFFERED=1
VOLUME ["/app/data", "/app/models"]
ENTRYPOINT ["footy"]
CMD ["bot"]

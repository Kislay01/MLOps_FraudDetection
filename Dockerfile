# One image for every Fraudar Python service; docker-compose.cloud.yml picks the command per service.
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src ./src
COPY scripts ./scripts
# data/ and models/ are mounted read-only at run time (pulled with `dvc pull` on the host)
RUN useradd --create-home fraudar && mkdir -p logs control && chown -R fraudar /app
USER fraudar

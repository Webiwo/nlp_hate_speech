FROM python:3.11-slim-bookworm

RUN apt-get update -y && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG GCP_KEY_FILE
COPY ${GCP_KEY_FILE} /app/credentials.json
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/credentials.json"

COPY . /app

ENV GOOGLE_APPLICATION_CREDENTIALS="/app/credentials.json"

RUN pip install --no-cache-dir -r /app/requirements.txt

RUN apt-get update && apt-get install -y curl apt-transport-https ca-certificates gnupg && \
    mkdir -p /usr/share/keyrings && \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
      | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
      > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get update && apt-get install -y google-cloud-cli && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/lib/google-cloud-sdk/bin:${PATH}"

EXPOSE 80

CMD ["uvicorn", "app_fast:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]
FROM python:3.11-slim-bookworm

RUN apt-get update -y && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 80

CMD ["uvicorn", "app_fast:app", "--host", "0.0.0.0", "--port", "80", "--workers", "1"]
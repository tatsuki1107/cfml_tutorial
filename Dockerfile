FROM python:3.9-slim-buster

WORKDIR /app
RUN apt-get update \
  && apt-get install -y gcc\
  && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade poetry \
  && poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock /app/

RUN poetry install

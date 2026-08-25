FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/*

COPY scenario.json /app/scenario.json
COPY astro_task.py /usr/local/bin/astro-task

RUN printf '%s\n' '{"actions":[]}' > /app/answer.json \
    && chmod 0666 /app/answer.json \
    && chmod 0444 /app/scenario.json \
    && chmod 0555 /usr/local/bin/astro-task

WORKDIR /app

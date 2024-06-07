FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Create dirs for:
# - Injecting config.yml: /root/.DANE
# - Mount point for input & output files: /data
# - Storing the source code: /src
RUN mkdir /root/.DANE /data /src /model

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install poetry==1.8.2

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 

WORKDIR /src    
# copy the pyproject file and install all the dependencies first
COPY ./pyproject.toml poetry.lock /src/
RUN --mount=type=cache,target=/home/.cache/pypoetry/cache \
    --mount=type=cache,target=/home/.cache/pypoetry/artifacts \
    poetry install --only main --no-interaction --no-ansi

# copy the rest into the source dir
COPY ./ /src

# Write provenance info about software versions to file
RUN echo "dane-visual-feature-extraction-worker;https://github.com/beeldengeluid/dane-visual-feature-extraction-worker/commit/$(git rev-parse HEAD)" >> /software_provenance.txt

ENTRYPOINT ["./docker-entrypoint.sh"]
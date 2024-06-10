FROM docker.io/python:3.10

# Create dirs for:
# - Injecting config.yml: /root/.DANE
# - Mount point for input & output files: /mnt/dane-fs
# - Storing the source code: /src
RUN mkdir /root/.DANE /mnt/dane-fs /src /model


WORKDIR /src

# copy the pyproject file and install all the dependencies first
RUN pip install --upgrade pip
RUN pip install poetry
COPY ./pyproject.toml /src
RUN --mount=type=cache,target=/home/.cache/pypoetry/cache \
    --mount=type=cache,target=/home/.cache/pypoetry/artifacts \
    poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi

# copy the rest into the source dir
COPY ./ /src

# create an objects dir in .git. This remains empty, only needs to be present for git rev to work
RUN mkdir /src/.git/objects  

# Write provenance info about software versions to file
RUN echo "dane-visual-feature-extraction-worker;https://github.com/beeldengeluid/dane-visual-feature-extraction-worker/commit/$(git rev-parse HEAD)" >> /software_provenance.txt

ENTRYPOINT ["./docker-entrypoint.sh"]

# NOTE:  RUN pip install --no-cache-dir torch should drop the image size

# OF: && pip cache purge (na poetry install command)

# syntax=docker/dockerfile:1.2

# COPY poetry.lock /
# RUN --mount=type=cache,target=/home/.cache/pypoetry/cache \
#     --mount=type=cache,target=/home/.cache/pypoetry/artifacts \
#     poetry install
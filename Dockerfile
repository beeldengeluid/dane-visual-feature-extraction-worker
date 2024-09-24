FROM docker.io/python:3.10.15@sha256:f5c258c1a5b0c9de773933cafce165d08d4fc52688dcd586544e9fdbf6be2c29 AS req

RUN python3 -m pip install pipx && \
  python3 -m pipx ensurepath

RUN pipx install poetry==1.7.1 && \
  pipx inject poetry poetry-plugin-export && \
  pipx run poetry config warnings.export false

COPY ./poetry.lock ./poetry.lock
COPY ./pyproject.toml ./pyproject.toml
RUN pipx run poetry export --format requirements.txt --output requirements.txt

FROM docker.io/python:3.10.15@sha256:f5c258c1a5b0c9de773933cafce165d08d4fc52688dcd586544e9fdbf6be2c29

# Create dirs for:
# - Injecting config.yml: /root/.DANE
# - Mount point for input & output files: /data
# - Storing the source code: /src
RUN mkdir \
  /data \
  /model \
  /root/.DANE \
  /src

WORKDIR /src

COPY --from=req ./requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./ /src

# Write provenance info about software versions to file
RUN echo "dane-visual-feature-extraction-worker;https://github.com/beeldengeluid/dane-visual-feature-extraction-worker/commit/$(git rev-parse HEAD)" >> /software_provenance.txt

ENTRYPOINT ["./docker-entrypoint.sh"]

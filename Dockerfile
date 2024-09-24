FROM docker.io/python:3.10.15@sha256:08a49662edbfff8a296e2db2cb3235e117d1caeb09c32cba65011a2de0306440 AS req

RUN python3 -m pip install pipx && \
  python3 -m pipx ensurepath

RUN pipx install poetry==1.7.1 && \
  pipx inject poetry poetry-plugin-export && \
  pipx run poetry config warnings.export false

COPY ./poetry.lock ./poetry.lock
COPY ./pyproject.toml ./pyproject.toml
RUN pipx run poetry export --format requirements.txt --output requirements.txt

FROM docker.io/python:3.10.15@sha256:08a49662edbfff8a296e2db2cb3235e117d1caeb09c32cba65011a2de0306440

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

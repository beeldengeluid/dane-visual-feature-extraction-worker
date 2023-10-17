FROM docker.io/python:3.10

# Create dirs for:
# - Injecting config.yml: /root/.DANE
# - Mount point for input & output files: /mnt/dane-fs
# - Storing the source code: /src
RUN mkdir /root/.DANE /mnt/dane-fs /src


WORKDIR /src

# copy the pyproject file and install all the dependencies first
RUN pip install --upgrade pip
RUN pip install poetry
COPY ./pyproject.toml /src
RUN poetry config virtualenvs.create false && poetry install --only main --no-interaction --no-ansi

# copy the rest into the source dir
COPY ./ /src

# Write provenance info about software versions to file
RUN echo "dane-visual-feature-extraction-worker;https://github.com/beeldengeluid/dane-visual-feature-extraction-worker/commit/$(git rev-parse HEAD)" >> /software_provenance.txt 

ENTRYPOINT ["./docker-entrypoint.sh"]
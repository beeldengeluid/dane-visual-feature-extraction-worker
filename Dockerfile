#FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime

#FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Create dirs for:
# - Injecting config.yml: /root/.DANE
# - Mount point for input & output files: /data
# - Storing the source code: /src
RUN mkdir /root/.DANE /data /src /model

WORKDIR /src    
# copy the pyproject file and install all the dependencies first
COPY ./requirements.txt /src/
RUN pip install -r requirements.txt

# copy the rest into the source dir
COPY ./ /src

# Write provenance info about software versions to file
RUN echo "dane-visual-feature-extraction-worker;https://github.com/beeldengeluid/dane-visual-feature-extraction-worker/commit/$(git rev-parse HEAD)" >> /software_provenance.txt

ENTRYPOINT ["./docker-entrypoint.sh"]
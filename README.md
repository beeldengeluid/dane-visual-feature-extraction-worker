# dane-visual-feature-extraction-worker
Apply VisXP models trained for feature extraction from keyframes and audio spectograms, to produce feature vectors.

## Docker  

### Installation 

From the root of the repo, run 
```
DOCKER_BUILDKIT=1 docker build -t dane-visual-feature-extraction-worker . 
```
Buildkit is optional, it may speed up building (see https://docs.docker.com/build/buildkit/)
NB: building the image has occasionally led to mysterious connection errors, which we haven't been able to track down and solve (yet). Discarding the poetry.lock has been a way to circumvent these. 

### Config

The following parts are relevant for local testing (without connecting to DANE). All defaults
are fine for testing, except:

- `VISXP_EXTRACT.TEST_INPUT_PATH`: make sure to supply your `.tar.gz` archive (for instance obtained through S3)
- `S3_ENDPOINT_URL`: ask your DANE admin for the endpoint URL
- `S3_BUCKET`: ask your DANE admin for the bucket name
NB: for S3 testing, you also need to supply valid S3 credentials, for instance through a s3-creds.env file. 

Optionally, add a model specification to the appropriate dir: `model/checkpoint.tar` and `model/model_config.yml`. 

If none is present, a model specification will be downloaded from S3, as indicated in the config: `MODEL_CHECKPOINT_S3_URI` and `MODEL_CONFIG_S3_URI`.

### Run test file in local Docker Engine

This form of testing/running avoids connecting to DANE:

- No connection to DANE RabbitMQ is made
- No connection to DANE ElasticSearch is made

This is ideal for testing:

- feature_extraction.py, which uses `VISXP_EXTRACT.TEST_INPUT_PATH` (see config.yml) to produce this worker's output
- I/O steps taken after the output is generated, i.e. deletion of input/output and transfer of output to S3

Run `docker-compose up` to run the worker in a container. By default, it will process the file specified in `config/config.yml`.

Check out the `docker-compose.yml` to learn about how the main process is started. As you can see there are two volumes mounted and an environment file is loaded:

```yml
version: '3'
services:
  web:
    image: dane-video-segmentation-worker:latest  # your locally built docker image
    volumes:
      - ./data:/data  # put input files in ./data and update VISXP_PREP.TEST_INPUT_FILE in ./config/config.yml
      - ./config:/root/.DANE  # ./config/config.yml is mounted to configure the main process
    container_name: visxp
    command: --run-test-file  # NOTE: comment this line to spin up th worker
    env_file:
      - s3-creds.env  # create this file with AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to allow boto3 to connect to your AWS S3 bucket (see OUTPUT.S3_* variables in config.yml)
    logging:
      options:
        max-size: 20m
    restart: unless-stopped
```

There is no need to update the docker-compose.yml, but make sure to:

- adapt `./config/config.yml` (see next sub-section for details)
- create `s3-creds.env` to allow the worker to upload output to your AWS S3 bucket
- tag your image to match the one mentioned in `docker-compose.yml` (`dane-visual-feature-extraction-worker` by default)

## Local environment

### Config 
Copy `config/config.yml` to the root of the repo: `./config.yml` as your local config file. 
In the (local) config file, replace all root directories (`/data` and `/model`) to actual directories on your local system (presumably `./data` and `./model`)
Add a model specification to the appropriate dir: `model/checkpoint.tar` and `model/model_config.yml`

### Installation 

From the root of the repo, run 
```sh
poetry install
```

### Testing
Run tests with: 
```sh
poetry run pytest 
```
Optionally add `--pdb` for debugging


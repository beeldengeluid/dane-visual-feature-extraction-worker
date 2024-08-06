# dane-visual-feature-extraction-worker
Apply VisXP models trained for feature extraction from keyframes and audio spectograms, to produce feature vectors.

## Docker  

### Installation 

From the root of the repo, run 
```
docker build -t dane-visual-feature-extraction-worker . 
```
NB: building the image has occasionally led to mysterious connection errors, which we haven't been able to track down and solve (yet). Discarding the poetry.lock has been a way to circumvent these. 

The Dockerfile support both CPU and GPU processing. Whether the processing actually uses GPU depends on the availability of GPU in the container. 

### Config

The following parts are relevant for local testing (without connecting to DANE). All defaults
are fine for testing, except:

- `VISXP_EXTRACT.TEST_INPUT_PATH`: make sure to supply your input (a `.tar.gz` archive or a directory with VISXP_PREP output, for instance obtained through S3) or a S3 location right away
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
    image: dane-visual-feature-extraction-worker:latest  # your locally built docker image
    volumes:
      - ./data:/data  # put input files in ./data and update VISXP_PREP.TEST_INPUT_FILE in ./config/config.yml
      - ./model:/model # mount the model dir so the model files are not required to be downloaded each time
      - ./config:/root/.DANE  # ./config/config.yml is mounted to configure the main process

    container_name: visxp_worker_2
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
Optionally, add a model specification to the appropriate dir: `model/checkpoint.tar` and `model/model_config.yml` (or specify the appropriate S3 URIs to have the worker download them).

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
Optionally add `--pdb` for debugging.
Also, optionally add `-m "not legacy"` to skip legacy tests (involving obsolete audio processing).
Some of the tests depend on model files that are private, which is why they are left out of the automated test pipeline in `.github/workflows/_test.yml`. These should be run locally when code under these tests is touched in a PR. 



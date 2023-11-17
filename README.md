# dane-visual-feature-extraction-worker
Apply VisXP models trained for feature extraction from keyframes and audio spectograms, to produce feature vectors.


## Installation:

Copy `config/config.yml` to the root of the repo: `./config.yml`.

Add a model specification to the appropriate dir: `model/checkpoint.tar` and `model/model_config.yml`


### Docker 

From the root of the repo, run 
```
DOCKER_BUILDKIT=1 docker build -t dane-visual-feature-extraction-worker . 
```
Buildkit is optional, it may speed up building (see https://docs.docker.com/build/buildkit/)
NB: building has occasionally led to mysterious connection errors, which we haven't been able to track down and solve (yet). 

### Local install
From the root of the repo, run 
```
Poetry install
```


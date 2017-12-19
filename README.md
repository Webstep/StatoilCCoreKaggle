## Webstep Repository for Kaggle Competition:  
### [Statoil/C-CORE Iceberg Classifier Challenge](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge)


Build Status: 
[![CircleCI](https://circleci.com/gh/milesgranger/StatoilCCoreKaggle/tree/master.svg?style=svg&circle-token=3fc7f5b381a1dd58d06545d0a1ccb71d96a96e3f)](https://circleci.com/gh/milesgranger/StatoilCCoreKaggle/tree/master)

## Getting started

This repository provides two solutions for setting up your environment: 
    1) docker or,
    2) local environment configuration

We recommend using docker because we will kept our latest environments up-to-date within the Dockerfiles.

#### Option 1) Docker:
Dockerfile(s) are found in `./docker`:

```commandline
    (CPU) milesg-cpu-Dockerfile

    (GPU) milesg-Dockerfile
```

If you are building the image for the first time, cd into the main directory of this repo and in your terminal execute either,

(CPU)
```commandline
docker-compose --file docker-compose-cpu.yml up
```

(GPU)
```commandline
docker-compose up
```

Note: Containers using nvidia-docker can be used with docker-compose by setting   
/etc/docker/daemon.json to the following:  
```json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": [ ]
        }
    }
}
```

The easiest way to launch our docker container is:

(CPU)
```commandline
docker run --rm -v $(pwd):/code -p 8888:8888 milesg/kaggle-icc-cpu jupyter notebook --allow-root --ip 0.0.0.0 --port 8888 --no-browser
```

(GPU)
```commandline
docker run --runtime=nvidia --rm -v $(pwd):/code -p 8888:8888 milesg/kaggle-icc jupyter notebook --allow-root --ip 0.0.0.0 --port 8888 --no-browser
```


#### Option 2) Local environment:

If you prefer not to use docker, there are two Anaconda environment files
you can use to create the same conda environment used in the docker image:

`conda-env-cpu.yml` & `conda-env-gpu.yml` for CPU and GPU environments, respectively.  

Create an environment by:  
```commandline
conda env create --file conda-env-cpu.yml --name icc-cpu
```

Update an environment with: 
```commandline
conda env update --file conda-env-cpu.yml --name icc-cpu
```

---

#### Data Loading:
Assumes data has been loaded into `data` directory from the competition page. 
If files are compressed in .7z format, and system has 7z installed, it will automatically be uncompressed.  

```python
from icc.data_loader import DataLoader
X, y = DataLoader.load_train()  # type: pd.DataFrame, pd.Series
X = DataLoader.load_test()    # type: pd.DataFrame
sample = DataLoader.load_sample_submission()  # type: pd.DataFrame
```

This data will be the EXACT same data used to pass to your model(s) within the StackedClassifier

---

#### Include your model in the ML Stack:

Reference `icc.models.example_model.py` for full example.

Your model MUST inherit from `sklearn.base.BaseEstimator` and implement the following:
- `fit(X, y)` -> fit your model and return `self`
- `predict(X)` -> return 1d array of predicted classes
- `predict_proba(X)` -> return array of shape [n_samples, 2] (probabilities of 0 and 1.. [[0.4, 0.6], ...])
- `get_params(deep=True)` -> return dict of parameters specifies in your model's `__init__` 


Example:
```python 

from icc.ml_stack import StackedClassifier


@StackedClassifier.register
class MySweetModel:

    def __init__(self):
        ...

    def fit(self, X, y):
        ...
        
    def predict(self, X):
        ...
        
    def predict_proba(self, X):
        ...
        
    def get_params(self, deep=True):
        ...
```

---


#### Running ML Stack

```python
from icc.ml_stack import run_stack
run_stack()
```

Or from the commandline:

```commandline
python icc/ml_stack.py
```

This outputs a `data/submission.csv` file ready for Kaggle.
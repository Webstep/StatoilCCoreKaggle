## Webstep Repository for Kaggle Competition:  
### [Statoil/C-CORE Iceberg Classifier Challenge](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge)

---
# Docs:

---

#### Docker: (if desired)
Dockerfile(s) are found in `./docker`  
Can be brought up with `docker-compose up`
* Containers using nvidia-docker can be used with docker-compose by setting   
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

---

#### Data Loading:
Assumes data has been loaded into `data` directory from the competition page. 
If files are compressed in .7z format, and system has 7z installed, it will automatically be uncompressed.  

```python
from icc.data_loader import DataLoader
train = DataLoader.load_train()  # type: Dict
test = DataLoader.load_test()    # type: Dict
sample = DataLoader.load_sample_submission()  # type: pandas.core.DataFrame
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
## This module holds all models

---

#### Requirements:  

Models should inherit from `sklearn.base.BaseEstimator`  
note that this does not bind one to scikit-learn, just that you should  
implement `fit(X, y)`, `predict(X)` and `predict_proba(X)` methods.  


Finally, to have model included, register the class definition as the following:  

```python
import numpy as np
from sklearn.base import BaseEstimator

# Local import
from icc.ml_stack import StackedClassifier


@StackedClassifier.register
class DumbModel(BaseEstimator):

    def fit(self, X, y):
        self.min = y.min()
        self.max = y.max()
        return self

    def predict(self, X):
        return np.random.randint(self.min, self.max, X.shape[0])
        
    def predict_proba(self, X):
        return np.random.random_sample(X.shape[0])
```
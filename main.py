import numpy as np
from custom_estimator import CustomEstimator
from custom_search import CustomSearch

# synthetic data
nb_features=5
nb_data_points=100
X_train = np.random.uniform(-1,+1,(nb_data_points,nb_features))
Y_train = (np.sign(X_train[:,0])).astype(np.int)
X_test = np.random.uniform(-1,+1,(nb_data_points,nb_features))
Y_test = (np.sign(X_test[:,0])).astype(np.int)


# create Estimator object
estimator=CustomEstimator()

# parameter the Searcher
search_parameters = {"search_parameter_min" : 1,
                "search_parameter_max": 2000,
                "search_parameter_sigma": 1000,
                "search_parameter_energy": 10,
                "search_parameter_iteration": 20
                }

# create Search object
searcher = CustomSearch(estimator=estimator, custom_search_param=search_parameters, verbose=True, cv=2,n_jobs=2)

# run Search algorithm
searcher.fit(X_train, y=Y_train)

# see best Estimator's parameter
print(searcher.best_params_)

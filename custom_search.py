#https://github.com/scikit-learn/scikit-learn/blob/ab9eebfa3a8d276de67b25ce42002a20298998b5/sklearn/model_selection/_search.py#L617
# BaseSearchCV begin to line
# GridSearchCV begin to line 1177

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.model_selection._search import BaseSearchCV
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class CustomSearch(BaseSearchCV):
    def __init__(self, estimator, custom_search_param,
                 scoring=None, fit_params=None, n_jobs=None, iid='warn',
                 refit=True, cv='warn', verbose=0, pre_dispatch='2*n_jobs',
                 error_score='raise-deprecating', return_train_score=True):
        BaseSearchCV.__init__(self,estimator, scoring=scoring,
                 fit_params=fit_params, n_jobs=n_jobs, iid=iid,
                 refit=refit, cv=cv, verbose=verbose, pre_dispatch=pre_dispatch,
                 error_score=error_score, return_train_score=return_train_score)
        print("custom_search_param="+str(custom_search_param))

        self.min=custom_search_param['search_parameter_min']
        self.max = custom_search_param['search_parameter_max']
        self.energy = custom_search_param['search_parameter_energy']
        self.iteration = custom_search_param['search_parameter_iteration']
        self.sigma = custom_search_param['search_parameter_sigma']

        self.cur_hyper_param = {'hyper_parameter': int(np.random.uniform( self.min, self.max ))}



    def proposal(self):
        cur=self.cur_hyper_param['hyper_parameter']

        prop_value=np.abs(int( np.random.normal(cur, self.sigma,) ))
        prop_value_with_constraints=np.min( [ np.max( [ prop_value , self.min ] ) , self.max ] )

        return dict({'hyper_parameter': prop_value_with_constraints})

    def manage_candidates(self,param_cadidates,score_candidates):#id=0 : proposal, id=1 self.cur_hyper_param
        loss = score_candidates[1] - score_candidates[0]
        accept_rate = np.exp(-self.energy * loss)
        accept = np.random.uniform(0., 1.) < accept_rate
        if accept:
            self.cur_hyper_param = param_cadidates[0]


        if self.verbose:
            print(".....")
            print("score_current="+str(score_candidates[1]) + " (hidden_layer="+str(param_cadidates[1]['hyper_parameter'])+")")
            print("score_proposed="+str(score_candidates[0]) + " (hidden_layer="+str(param_cadidates[0]['hyper_parameter'])+")")
            print("accept_rate=" + str(accept_rate))
            print("accept ? "+str(accept))
            print(".....")

    def _run_search(self,evaluate_candidates):

        for i in range(self.iteration):
            prop_hyper_param=self.proposal()

            param_candidates = [prop_hyper_param,self.cur_hyper_param]
            all_results = evaluate_candidates(param_candidates) # can be optimized
            score_candidates = all_results['mean_test_score'][-1*len(param_candidates) : ]

            if self.verbose:
                print("ITERATION="+str(i))
            self.manage_candidates(param_candidates,score_candidates)

# scikit-learn-customization
How implement your own scikit learn algorithm

<h1> Why your algorithm should be scikit-learn objects ? </h1>
<ul>
<li> Because your algorithm could naturally use scikit-learn objects and be compared easily with existing algorithms. </li>
<li> Because it constraints you to use their architecture development : structure choice have already done by the dedicated scikit-learn team. </li>
<li> Because a lot of project use scikit-learn object in parameters and can benefit to your project. Like <a href="https://github.com/slundberg/shap"> shap </a> (interpretability) and <a href="https://github.com/dask/dask">dask</a> (parallel computing). </li>
</ul>

<h1> How scikit learn objects work ? </h1>
<ul>
<li> Estimator (inherit from BaseEstimator and ClassifierMixin) : this object contains Machine Learning algorithm. In this tiny example (CustomEstimator) is a multi-layer peceptron and we want discover the best number of neurons. </li>
<li> Searcher (inherit from BaseSearchCV) : this object contains algorithm to explore hyper-parameter space and find best solution as possible. In this tiny example (CustomSearch) implements simulated algorithm algorithm. </li>
</ul>

<h1> Requirements </h1>
<ul>
<li/> Scikit-learn >= 0.20
</ul>

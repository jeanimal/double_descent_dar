# double_descent_dar

Reproducing the ideas in a simple way from this wonderful but highly mathematical paper:

Yehuda Dar, Muthukumar, V., & Baraniuk, R. (2021). A Farewell to the
Bias-Variance Tradeoff? An Overview of the Theory of Overparameterized
Machine Learning. https://arxiv.org/abs/2109.02355

Notebooks:
- Focus on double descent in under vs. over-parametered models: https://github.com/jeanimal/double_descent_dar/blob/main/double_descent_mtcars_reg.ipynb
- Double descent using gradient descent to estimate the linear regression: https://github.com/jeanimal/double_descent_dar/blob/main/double_descent_by_gradient_descent.ipynb

Libraries:
- `Sample_eval`, a wrapper to sklearn functions to make it easier to control under vs. over-parameterization when training: https://github.com/jeanimal/double_descent_dar/tree/main/double_descent_dar 

Fully explained in:

- Blog post: (new post coming but for now...)
  https://medium.com/@jeanimal/technical-even-linear-regression-can-escape-the-bias-variance-tradeoff-263abe6acb1c
- Video: https://youtu.be/bM6WJVyytEg

More Code:

- This code reproduction is in python. A reproduction in R is here:
  https://github.com/jeanimal/farewell_bias_variance

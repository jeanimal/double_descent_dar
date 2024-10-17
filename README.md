# double_descent_dar

Reproducing and playing with some ideas using code from this wonderful but highly mathematical paper:

Yehuda Dar, Muthukumar, V., & Baraniuk, R. (2021). A Farewell to the
Bias-Variance Tradeoff? An Overview of the Theory of Overparameterized
Machine Learning. https://arxiv.org/abs/2109.02355

## Blog

The best summary of what I explore in this repo and WHY is in my ***glog***:

https://medium.com/@jeanimal/how-double-descent-breaks-the-shackles-of-the-interpolation-threshold-5f5990dc02b2

But... it doesn't cover everything here yet.

## In this repo

Notebooks:
- Manipulating double descent by manipulating the parameterization ratio-- directly relevant to the blog post: https://github.com/jeanimal/double_descent_dar/blob/main/double_descent_mtcars_reg.ipynb
- Double descent using **gradient descent** to estimate the linear regression-- going beyond the blog post: https://github.com/jeanimal/double_descent_dar/blob/main/double_descent_by_gradient_descent.ipynb
- Generating tables to show underparameterized and overparameterized fits (for my blog but not very interesting): https://github.com/jeanimal/double_descent_dar/blob/main/Fitting%20overparameterized%20linear%20models.ipynb

Libraries:
- `Sample_eval`, a wrapper to sklearn functions to make it easier to control under vs. over-parameterization when training: https://github.com/jeanimal/double_descent_dar/tree/main/double_descent_dar 

## Give me more!

Older explanations of double descent, focusing more on the necessary structure of the data:

- Blog post: (new post coming but for now...)
  https://medium.com/@jeanimal/technical-even-linear-regression-can-escape-the-bias-variance-tradeoff-263abe6acb1c
- Video: https://youtu.be/bM6WJVyytEg

More Code:

- This code  is in python. A reproduction of the Dar paper in R is here:
  https://github.com/jeanimal/farewell_bias_variance

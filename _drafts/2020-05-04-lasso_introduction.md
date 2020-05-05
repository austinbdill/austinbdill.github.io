---
layout: post
title: "LASSO #1: Introduction"
description: Examining the motivation for the LASSO problem and the concept of sparsity
author: Austin Dill
bib: test.bib
date: 2020-05-04
---
### Introduction

In this blog series, I'm going to be discussing the LASSO problem for linear regression, including its motivation, theoretical guarantees, and computation. This post will focus on the shortcomings of classical ordinary least squares regression with a focus on the concept of "sparsity".

Let's consider the task of fitting a linear function of $$m$$ features to a scalar output for $$n$$ examples in the presence of i.i.d. noise. We'll represent this with a matrix of the $$n$$ examples stacked on top of eachother, $$X \in \mathcal{R}^{n\times m}$$, the target vector $$y \in \mathcal{R}^n$$, and the coefficient vector $$\beta^* \in \mathcal{R}^m$$. 

<div style="text-align: center">
    <script type='math/tex; mode=display'>
        y = X\beta^* + \epsilon
    </script>
</div>

Our goal is to find a set of coefficients that closely approximate the relationship $$ y \approx X\beta $$.

Without loss of generality, we will also make the following assumptions about our design matrix $$X$$:

- The columns are linearly independent.
- Each feature has mean zero.
- Each column has length one. 

These assumptions will make the math more interpretable without changing our results. 

{% highlight python %}

def generate_example(n, p, s, low=-1, high=1, sigma=0.01):
    #Generate dense coefficient vector
    beta_gt = np.random.uniform(low=low, high=high, size=p)
    
    #Create sparse coefficient vector
    indices = np.random.choice(np.arange(p), replace=False, size=p-s)
    beta_gt[indices] = 0
    
    #Create X matrix
    X = np.random.normal(size=(n, p))
    X = X - np.mean(X, axis=0)
    X = X / np.linalg.norm(X, axis=0)
    
    
    #Create noiseless y vector
    y = X@beta_gt
    
    #Create noisy y vector
    y = y + np.random.normal(scale=sigma, size=(n,))
    y = y - y.mean()
    
    return X, y, beta_gt

{% endhighlight %}

### Ordinary Least Squares

The classical approach to this problem is to penalize distance between the target and the prediction in terms of the $$\ell_2$$ norm. The produces the *Ordinary Least Squares* regression problem shown below. 

<div style="text-align: center">
    <script type='math/tex; mode=display'>
        \min_\beta \|y-X\beta\|_2^2
    </script>
</div>

Differentiating and setting the gradient equal to zero we get the familiar *normal equations*. 

<div style="text-align: center">
    <script type='math/tex; mode=display'>
        X^TX\beta = X^Ty
    </script>
</div>

While this provides a simple, closed-form solution to the regression problem, we'll see that the ordinary least squares (OLS) solution fails to capture some of the true underlying structure of the true coefficient vector. 

{% highlight python %}

def solve_ols(X, y):
    return np.linalg.solve(X.T@X, X.T@y)

{% endhighlight %}

### Sparsity

Suppose that $$\beta^*$$ is only nonzero on a subset of indices $$S = \{ i_1, \dots, i_s\}$$ called the support of $$\beta^*$$. When $$s << m$$ we say that $$\beta^*$$ is sparse. 

One reason for seeking a sparse solution is that it is more interpretable. If you're given a problem with a thousand variables but only two are relevant, it'll be much easier to understand the relationship between variables. As we'll discuss in future blog posts, sparsity also comes with a number of theoretical advantages especially in high dimensional spaces.

In addition to wanting an estimator with a low approximation error, it is natural to see estimators that discover the true underlying support set. This property is sometimes known as *sparsistence*. 

#### Sparsistence

**Definition:** With high probability, our estimate $$\hat{\beta}$$ has the same support as the ground truth vector $$\beta^*$$. 

#### OLS Failure

### Best Subset Selection

### The LASSO

### Conclusion

---
#### References

{% bibliography --cited %} 



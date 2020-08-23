# Training the model

`neonrvm_train` function requires a pair of training parameter structures, one
for just getting rid of pretty useless basis functions, and another one for
polishing the training results. In the first one you usually want to keep
majority of basis functions, while in the last one you want to reduce number of
basis functions as much as possible in order to achieve a more general and
sparse model.

By choosing a *batch size* value smaller than total input basis function count,
the model will be trained incrementally using the first training parameter
structure, and will get polished using the last training parameter structure in
the end.

neonrvm has been carefully designed to handle multiple training scenarios.
Different scenarios are discussed below:

## A) Optimizing kernel parameters

During this period one needs to rapidly try different kernel parameters either
using an optimization algorithm ([Hyperopt] to the rescue) or brute force method
to achieve optimal kernel parameters. Make sure that the optimization algorithm
can deal with possible training failures.

In this case users need to use small batch sizes, and training parameters with
more relaxed convergence conditions. In other words, a highly polished and
sparse model isn't required.

## B) Finalized kernel parameters and model

When you are finished with tuning kernel parameters and trying different model
creation ideas, you need access to the best basis functions and finely tuned
weights associated to them in order to make accurate predictions.

You need to throw bigger training batch sizes at neonrvm, and use training
parameters with low basis function percentage and high iteration count for the
polishing step in this case.

## 🍔) Big data sets

Memory and storage requirements do quickly skyrocket when dealing with large
data sets. You don't necessarily need to feed the whole design matrix to the
neonrvm all at once. It can also be fed in smaller chunks by loading different
design matrix parts from disk, or simply generating them on the fly.

neonrvm allows users to split the design matrix and perform the training process
incrementally at higher level through caching mechanism provided. You just need
to make multiple `neonrvm_train` function calls and neonrvm will store the
useful basis functions in the given `neonrvm_cache` on the go.

It is a good idea to group together similar data using [clustering] algorithms,
and feed neonrvm with a mixture of them incrementally.

## 🚀 C/C++

Alright, now that we covered the different use cases, it's time to get familiar
with the `neonrvm_train` function:

```C
int neonrvm_train(neonrvm_cache* cache, neonrvm_param* param1, neonrvm_param* param2,
                  double* phi, size_t* index, size_t count, size_t batch_size_max)
```

➡️ *Parameters*

- ⬇️ `[in] cache`: Stores intermediate variables and training results.
- ⬇️ `[in] param1`: Incremental training parameters.
- ⬇️ `[in] param2`: Final polish parameters.
- ⬇️ `[in] phi`: Column major design matrix, with row count equivalent to the
    total sample count, and column count equivalent to the given basis function
    *count*. A copy of useful basis functions will be kept inside the training
    *cache*.
- ⬇️ `[in] index`: Basis function indices, a vector with elements count
    equivalent to the given basis function *count*. A copy of useful basis
    function indices will be kept inside the training *cache*. Must be a vector
    of positive numbers, and shouldn't contain any value equal to the
    `SIZE_MAX`, which is used internally to identify bias index.
- ⬇️ `[in] count`: Number of basis functions given. Must be a positive non-zero
    number.
- ⬇️ `[in] batch_size_max`: Maximum number of basis functions in every
    incremental training session. Must be a positive non-zero value.

⬅️ *Returns*

- `NEONRVM_SUCCESS`: After successful execution.
- `NEONRVM_INVALID_Px`: When facing erroneous parameters.
- `NEONRVM_LAPACK_ERROR`: When LAPACK fails to perform factorization or solve
    equations.
- `NEONRVM_MATH_ERROR`: When `NaN` or `∞` numbers show up in the calculations.

## 🐍 Python

```Python
def train(cache: Cache, param1: Param, param2: Param,
          phi: numpy.ndarray, index: numpy.ndarray, batch_size_max: int)
```

➡️ *Parameters*

- ⬇️ `[in] cache`: Stores intermediate variables and training results.
- ⬇️ `[in] param1`: Incremental training parameters.
- ⬇️ `[in] param2`: Final polish parameters.
- ⬇️ `[in] phi`: Column major design matrix, with row count equivalent to the
    total sample count, and column count equivalent to the given basis function
    *count*. A copy of useful basis functions will be kept inside the training
    *cache*.
- ⬇️ `[in] index`: Basis function indices, a vector with elements count
    equivalent to the given basis function *count*. A copy of useful basis
    function indices will be kept inside the training *cache*. Must be a vector
    of positive numbers, and shouldn't contain any value equal to the
    `SIZE_MAX`, which is used internally to identify bias index.
- ⬇️ `[in] batch_size_max`: Maximum number of basis functions in every
    incremental training session. Must be a positive non-zero value.

⬅️ *Returns*

- Nothing that I'm aware of.

[Hyperopt]: https://github.com/hyperopt/hyperopt
[clustering]: https://en.wikipedia.org/wiki/Cluster_analysis

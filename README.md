# Introduction

**neonrvm** is an experimental open source machine learning library for 
performing regression tasks using [RVM] technique. It is written in C 
programming language and comes with bindings for the Python programming 
language.

Under the hood neonrvm uses expectation maximization fitting method, and allows 
basis functions to be fed incrementally to the model. This helps to keep training 
times and memory requirements significantly lower for large data sets.

neonrvm is not trying to be a full featured machine learning framework, and only 
provides core training and prediction facilities. You might want to use it in 
conjunction with higher level scientific programming languages and machine 
learning tool kits instead.

[RVM]: https://en.wikipedia.org/wiki/Relevance_vector_machine

---

# Building neonrvm

You can build neonrvm as a dynamic or static library; or manually include 
`neonrvm.h` and `neonrvm.c` in your C/C++ project and handle linkage of the 
required dependencies.

A C99 compiler is required in order to compile neonrvm, you can find one in 
every house these days. You also need [CMake] to configure and generate build 
files.

neonrvm requires a linear algebra library providing CBLAS/LAPACKE 
interface to do its magic. Popular ones are [Intel MKL], [OpenBLAS], and 
the reference [Netlib LAPACKE].

Python bindings can be installed from the source package using [Flit] Python 
module by simply running:
```Shell
flit install
```

or from [PyPI] software repository using following command:
```Shell
pip install neonrvm
```

[CMake]: https://cmake.org/
[Intel MKL]: https://software.intel.com/mkl
[OpenBLAS]: http://www.openblas.net/
[Netlib LAPACKE]: http://www.netlib.org/lapack/
[Flit]: https://github.com/takluyver/flit
[PyPI]: https://pypi.python.org/pypi

---

# Using neonrvm

Congratulations, you survived the build process! Following are general tips 
and steps in order to train your model and perform predictions using neonrvm. 
Please have a look at `example.c` and `example.py` for working sample codes. 
At this point it's a good idea to grab the original RVM paper and other related 
papers to get a feeling of inner workings of the RVM technique and different 
parameters.

In order to keep repetitions in this document lower, Python bindings are 
briefly documented. Errors reported by the library, will be raised as 
exceptions in Python.

[Sparse Bayesian Models (and the RVM)](http://www.miketipping.com/sparsebayes.htm)

## Step 0: Data preparation

This is the most important step in machine learning, and performance of your 
model totally depends on it. Some general tips includes:
- Cleaning your data set from suspicious and wrong data
- Feature engineering and giving more hints to the model
- Normalizing and scaling input data
- Randomizing input data order

There is definitely more to that list, and it's strongly recommended to spend 
more time on studying and preparation of input data than selection and tweaking 
of the machine learning method. [pandas] and [scikit-learn] are your best 
friends for data preparation if you are familiar with [Python] programming 
language.

[pandas]: https://pandas.pydata.org/
[scikit-learn]: http://scikit-learn.org
[Python]: https://www.python.org/

## Step 1: Design matrix preparation

Design matrix is a 2D matrix with `m` rows and `m` columns, with `m` being 
equivalent to the number of input data samples. There is usually a column 
consisting of `1.0` appended to that matrix to account for [bias], which makes 
it `m*(m+1)` or `m*n`, with `n` representing the number of [basis functions].

In plain English, basis functions show us how much close and similar input data 
are to each other. And a [kernel function] decides how much similar our input 
data are. Selection of the kernel function depends on the problem at hand, and 
you can even mix multiple kernel functions together.

An [RBF] kernel with suitable parameter usually gives satisfactory results. 
Our old buddy scikit-learn is there again to help you with a wide selection of 
kernel functions and optimizing their parameters.

Before passing your design matrix to the neonrvm, make sure that it's stored in 
[column major] order in memory. neonrvm will automatically append an extra 
column for bias to the design matrix during *training* process, so you just 
need to prepare a 2D `m*m` matrix.

[bias]: https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff
[basis functions]: https://en.wikipedia.org/wiki/Basis_function
[kernel function]: https://en.wikipedia.org/wiki/Kernel_method
[RBF]: https://en.wikipedia.org/wiki/Radial_basis_function
[column major]: https://en.wikipedia.org/wiki/Row-_and_column-major_order

## Step 2: Creating training cache

`neonrvm_cache` structure acts as a cache for storing a couple of intermediate 
training results and allows us to reuse memory as much as possible during 
learning process.

### C/C++

You can create one using `neonrvm_create_cache` function described below:

```C
int neonrvm_create_cache(neonrvm_cache** cache, double* y, size_t count)
```

*Parameters*  
- `[out] cache`: Pointer which it points to will be set to a freshly allocated 
    structure.
- `[in] y`: Data set output/target array, a copy will be made of its contents.
- `[in] count`: `y` array elements count.

*Returns*  
- `NEONRVM_SUCCESS`: After successful execution.
- `NEONRVM_INVALID_Px`: When facing erroneous parameters.

Once you are done with `neonrvm_cache` structure and finished training process, 
you should call `neonrvm_destroy_cache` to free up allocated memory.

```C
int neonrvm_destroy_cache(neonrvm_cache* cache)
```

*Parameters*  
- `[in] cache`: Memory allocated for this structure will be released.

*Returns*  
- `NEONRVM_SUCCESS`: After successful execution.
- `NEONRVM_INVALID_Px`: When facing erroneous parameters.

### Python

You simply need to create a new `Cache` instance, no need for manual memory 
management.

```Python
class Cache(y: numpy.ndarray)
```

*Returns*  
- A new `Cache` instance.

## Step 3: Creating training parameters

`neonrvm_param` structure deals with training convergence conditions and 
initial values.

### C/C++

Use `neonrvm_create_param` function to create one:

```C
int neonrvm_create_param(neonrvm_param** param, 
                         double alpha_init, double alpha_max, double alpha_tol, 
                         double beta_init, double basis_percent_min, size_t iter_max)
```

*Parameters*  
- `[out] param`: Pointer which it points to will be set to a freshly allocated 
    structure.
- `[in] alpha_init`: Initial value for *alpha*. Must be a positive and small 
    number.
- `[in] alpha_max`: Basis functions associated with *alpha* value beyond this 
    limit will be purged. Must be a positive and big number.
- `[in] alpha_tol`: Training session will end if changes in *alpha* values gets 
    lower than this value. Must be a positive and small number.
- `[in] beta_init`: Initial value for *beta*. Must be a positive and small value.
- `[in] basis_percent_min`: Training session will end if percentage of useful 
    basis functions during current training session gets lower than this value. 
    Must be a value in `[0.0, 100.0]` range.
- `[in] iter_max`: Training session will end if training loop iteration count 
    goes beyond this value. Must be a positive and non-zero number.

*Returns*  
- `NEONRVM_SUCCESS`: After successful execution.
- `NEONRVM_INVALID_Px`: When facing erroneous parameters.

Once you are done with `neonrvm_param` structure and finished training process, 
you should call `neonrvm_destroy_param` to free up allocated memory.

```C
int neonrvm_destroy_param(neonrvm_param* param)
```

*Parameters*  
- `[in] param`: Memory allocated for this structure will be released.

*Returns*  
- `NEONRVM_SUCCESS`: After successful execution.
- `NEONRVM_INVALID_Px`: When facing erroneous parameters.

### Python

A new `Param` instance should be created:

```Python
class Param(alpha_init: float, alpha_max: float, alpha_tol: float,
            beta_init: float, basis_percent_min: float, iter_max: int)
```

*Returns*  
- A new `Param` instance.

## Step 4: Training the model

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

### A) Optimizing kernel parameters

During this period one needs to rapidly try different kernel parameters either 
using an optimization algorithm ([SciPy] to the rescue) or brute force method 
to achieve optimal kernel parameters.

In this case users need to use small batch sizes, and training parameters with 
more relaxed convergence conditions. In other words, a highly polished and 
sparse model isn't required.

### B) Finalized kernel parameters and model

When you are finished with tuning kernel parameters and trying different model 
creation ideas, you need access to the best basis functions and finely tuned 
weights associated to them in order to make accurate predictions.

You need to throw bigger training batch sizes at neonrvm, and use training 
parameters with low basis function percentage and high iteration count for the 
polishing step in this case.

### 🍔) Big data sets

Memory and storage requirements do quickly skyrocket when dealing with large 
data sets. You don't necessarily need to feed the whole design matrix to the 
neonrvm all at once. It can also be fed in smaller chunks by loading different 
design matrix parts from disk, or simply generating them on the fly.

neonrvm allows users to split the design matrix and perform the training 
process incrementally at higher level through caching mechanism provided. You 
just need to make multiple `neonrvm_train` function calls and neonrvm will 
store the useful basis functions in the given `neonrvm_cache` on the go.

### C/C++

Alright, now that we covered the different use cases, it's time to get familiar 
with the `neonrvm_train` function:

```C
int neonrvm_train(neonrvm_cache* cache, neonrvm_param* param1, neonrvm_param* param2, 
                  double* phi, size_t* index, size_t count, size_t batch_size_max)
```

*Parameters*  
- `[in] cache`: Stores intermediate variables and training results.
- `[in] param1`: Incremental training parameters.
- `[in] param2`: Final polish parameters.
- `[in] phi`: Column major design matrix, with row count equivalent to the 
    total sample count, and column count equivalent to the given basis function 
    *count*. A copy of useful basis functions will be kept inside the training 
    *cache*.
- `[in] index`: Basis function indices, a vector with elements count equivalent 
    to the given basis function *count*. A copy of useful basis function indices 
    will be kept inside the training *cache*. Must be a vector of positive 
    numbers, and shouldn't contain any value equal to the `SIZE_MAX`, which is 
    used internally to identify bias index.
- `[in] count`: Number of basis functions given. Must be a positive non-zero 
    number. Total number of basis functions shouldn't exceed the total number 
    of samples.
- `[in] batch_size_max`: Maximum number of basis functions in every incremental 
    training session. Must be a positive non-zero value.

*Returns*  
- `NEONRVM_SUCCESS`: After successful execution.
- `NEONRVM_INVALID_Px`: When facing erroneous parameters.
- `NEONRVM_LAPACKE_ERROR`: When LAPACKE fails to perform factorization or solve 
    equations.
- `NEONRVM_MATH_ERROR`: When `NaN` or `∞` numbers show up in the calculations.

### Python

```Python
def train(cache: Cache, param1: Param, param2: Param,
          phi: numpy.ndarray, index: numpy.ndarray, batch_size_max: int)
```

*Returns*  
- Nothing that I'm aware of.

[SciPy]: https://www.scipy.org/

## Step 5: Getting the training results

After successful completion of training process, training results including 
useful basis function indices and their associated weights can be queried using 
`neonrvm_get_training_stats` and `neonrvm_get_training_results` functions.

You should first get the useful basis functions count, and then allocate enough 
memory for the basis function indices and weights vectors so neonrvm can fill 
them for you.

### C/C++

```C
int neonrvm_get_training_stats(neonrvm_cache* cache, size_t* basis_count, bool* bias_used)
```

*Parameters*  
- `[in] cache`: Contains intermediate variables and training results.
- `[out] basis_count`: Value pointed to will be set to the number of useful 
    basis functions. (Includes *bias* too if it was found useful)
- `[out] bias_used`: Value pointed to will be set to `true` if *bias* was 
    useful during training.

*Returns*  
- `NEONRVM_SUCCESS`: After successful execution.
- `NEONRVM_INVALID_Px`: When facing erroneous parameters.

```C
int neonrvm_get_training_results(neonrvm_cache* cache, size_t* index, double* mu)
```

*Parameters*  
- `[in] cache`: Contains intermediate variables and training results.
- `[out] index`: Vector with enough room for useful basis function indices. 
    Last element contains `SIZE_MAX` if *bias* was found to be useful.
- `[out] mu`: Vector with enough room for useful basis function weights. Last 
    element contains *bias* weight, if *bias* was found to be useful.

*Returns*  
- `NEONRVM_SUCCESS`: After successful execution.
- `NEONRVM_INVALID_Px`: When facing erroneous parameters.

### Python

A single function call is enough:

```Python
def get_training_results(cache: Cache)
```

*Returns*  
- `index`: `numpy.ndarray`
- `mu`: `numpy.ndarray`
- `basis_count`: `int`
- `bias_used`: `bool`

## Step 6: Profit!

Now that you have the indices and weights of the useful data in hand, you can 
make predictions. In order to make prediction with new input data and unknown 
outcomes, you should create a new design matrix like what was discussed in 
*Step 1*, but this time getting populated with closeness and similarities 
between new input data, and useful data which we found their indices previously.

If *bias* was found useful during training process, you need to manually append 
a column of `1.0` to your new matrix. The new matrix should have row count 
equal to the number of new input data samples, and column count equal to the 
number of useful basis functions.

Prediction are made simply by multiplying the result matrix and weights vector. 
Output vector with a lenghts equal to the number of new input data samples 
contains the prediction outcomes.

### C/C++

You can use the `neonrvm_predict` function to make predictions.

```C
int neonrvm_predict(double* phi, double* mu,
                    size_t sample_count, size_t basis_count, double* y)
```

*Parameters*  
- `[in] phi`: Column major matrix, with row count equivalent to the 
    *sample_count*, and column count equivalent to the *basis_count*.
- `[in] mu`: Vector of associated basis function weights, with number of 
    elements equal to the *basis_count*.
- `[in] sample_count`: Number of input data samples with unknown outcomes.
- `[in] basis_count`: Number of useful basis functions.
- `[out] y`: Vector of predictions made for input data with unknown outcomes, 
    with enough room and number of elements equal to the *sample_count*.

*Returns*  
- `NEONRVM_SUCCESS`: After successful execution.
- `NEONRVM_INVALID_Px`: When facing erroneous parameters.
- `NEONRVM_MATH_ERROR`: When `NaN` or `∞` numbers show up in the calculations.

### Python

Number of `phi` columns and `mu` length should match.

```Python
def predict(phi: np.ndarray, mu: np.ndarray)
```

*Returns*  
- `y`: `numpy.ndarray`

---

# License

- neonrvm is licensed under the [MIT] license. Please see `LICENSE` for more 
    details.

- neonrvm includes code from [Netlib LAPACK] library, which is licensed under 
    a [modified BSD license].

- The relevance vector machine is [patented] in the United States by [Microsoft].

[MIT]: https://en.wikipedia.org/wiki/MIT_License
[Netlib LAPACK]: http://www.netlib.org/lapack/
[modified BSD license]: http://www.netlib.org/lapack/LICENSE.txt
[patented]: https://patents.google.com/patent/US6633857
[Microsoft]: https://www.microsoft.com


---

# Future work (Patches are welcome)

- Investigate methods to make learning process numerically more stable
- Implement classification
- Create higher level wrappers and programming language bindings
- Improve documentation

---

# Reference

- Tipping, M. E. (2001). Sparse Bayesian learning and the relevance vector machine. Journal of machine learning research, 1(Jun), 211-244.
- Ben-Shimon, D., & Shmilovici, A. (2006). Accelerating the relevance vector machine via data partitioning. Foundations of Computing and Decision Sciences, 31(1), 27-42.

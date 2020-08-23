# Getting the training results

After successful completion of training process, training results including
useful basis function indices and their associated weights can be queried using
`neonrvm_get_training_stats` and `neonrvm_get_training_results` functions.

You should first get the useful basis functions count, and then allocate enough
memory for the basis function indices and weights vectors so neonrvm can fill
them for you.

## 🚀 C/C++

```C
int neonrvm_get_training_stats(neonrvm_cache* cache, size_t* basis_count, bool* bias_used)
```

➡️ *Parameters*

- ⬇️ `[in] cache`: Contains intermediate variables and training results.
- ⬆️ `[out] basis_count`: Value pointed to will be set to the number of useful
    basis functions. (Includes *bias* too if it was found useful)
- ⬆️ `[out] bias_used`: Value pointed to will be set to `true` if *bias* was
    useful during training.

⬅️ *Returns*

- `NEONRVM_SUCCESS`: After successful execution.
- `NEONRVM_INVALID_Px`: When facing erroneous parameters.

```C
int neonrvm_get_training_results(neonrvm_cache* cache, size_t* index, double* mu)
```

➡️ *Parameters*

- ⬇️ `[in] cache`: Contains intermediate variables and training results.
- ⬆️ `[out] index`: Vector with enough room for useful basis function indices.
    Last element contains `SIZE_MAX` if *bias* was found to be useful.
- ⬆️ `[out] mu`: Vector with enough room for useful basis function weights. Last
    element contains *bias* weight, if *bias* was found to be useful.

⬅️ *Returns*

- `NEONRVM_SUCCESS`: After successful execution.
- `NEONRVM_INVALID_Px`: When facing erroneous parameters.

## 🐍 Python

A single function call is enough:

```Python
def get_training_results(cache: Cache)
```

➡️ *Parameters*

- ⬇️ `[in] cache`: Contains intermediate variables and training results.

⬅️ *Returns*

- `index: numpy.ndarray`: Vector of useful basis function indices. Last element
  contains `SIZE_MAX` if *bias* was found to be useful.
- `mu: numpy.ndarray`: Vector of useful basis function weights. Last element
  contains *bias* weight, if *bias* was found to be useful.
- `basis_count: int`: Number of useful basis functions. (Includes *bias* too if
  it was found useful)
- `bias_used: bool`: Whether *bias* was useful during training.

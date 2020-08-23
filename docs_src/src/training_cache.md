# Creating training cache

`neonrvm_cache` structure acts as a cache for storing a couple of intermediate
training results and allows us to reuse memory as much as possible during
learning process.

## 🚀 C/C++

You can create one using `neonrvm_create_cache` function described below:

```C
int neonrvm_create_cache(neonrvm_cache** cache, double* y, size_t count)
```

➡️ *Parameters*

- ⬆️ `[out] cache`: Pointer which it points to will be set to a freshly
    allocated structure.
- ⬇️ `[in] y`: Data set output/target array, a copy will be made of its
  contents.
- ⬇️ `[in] count`: `y` array elements count.

⬅️ *Returns*

- `NEONRVM_SUCCESS`: After successful execution.
- `NEONRVM_INVALID_Px`: When facing erroneous parameters.

Once you are done with `neonrvm_cache` structure and finished training process,
you should call `neonrvm_destroy_cache` to free up allocated memory.

```C
int neonrvm_destroy_cache(neonrvm_cache* cache)
```

➡️ *Parameters*

- ⬇️ `[in] cache`: Memory allocated for this structure will be released.

⬅️ *Returns*

- `NEONRVM_SUCCESS`: After successful execution.
- `NEONRVM_INVALID_Px`: When facing erroneous parameters.

## 🐍 Python

You simply need to create a new `Cache` instance, no need for manual memory
management.

```Python
class Cache(y: numpy.ndarray)
```

➡️ *Parameters*

- ⬇️ `[in] y`: Data set output/target array, a copy will be made of its
  contents.

⬅️ *Returns*

- A new `Cache` instance.

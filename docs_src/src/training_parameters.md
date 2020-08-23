# Creating training parameters

`neonrvm_param` structure deals with training convergence conditions and initial
values.

## 🚀 C/C++

Use `neonrvm_create_param` function to create one:

```C
int neonrvm_create_param(neonrvm_param** param,
                         double alpha_init, double alpha_max, double alpha_tol,
                         double beta_init, double basis_percent_min, size_t iter_max)
```

➡️ *Parameters*

- ⬆️ `[out] param`: Pointer which it points to will be set to a freshly
    allocated structure.
- ⬇️ `[in] alpha_init`: Initial value for *alpha*. Must be a positive and small
    number.
- ⬇️ `[in] alpha_max`: Basis functions associated with *alpha* value beyond this
    limit will be purged. Must be a positive and big number.
- ⬇️ `[in] alpha_tol`: Training session will end if changes in *alpha* values
    gets lower than this value. Must be a positive and small number.
- ⬇️ `[in] beta_init`: Initial value for *beta*. Must be a positive and small
  value.
- ⬇️ `[in] basis_percent_min`: Training session will end if percentage of useful
    basis functions during current training session gets lower than this value.
    Must be a value in `[0.0, 100.0]` range.
- ⬇️ `[in] iter_max`: Training session will end if training loop iteration count
    goes beyond this value. Must be a positive and non-zero number.

⬅️ *Returns*

- `NEONRVM_SUCCESS`: After successful execution.
- `NEONRVM_INVALID_Px`: When facing erroneous parameters.

Once you are done with `neonrvm_param` structure and finished training process,
you should call `neonrvm_destroy_param` to free up allocated memory.

```C
int neonrvm_destroy_param(neonrvm_param* param)
```

➡️ *Parameters*

- ⬇️ `[in] param`: Memory allocated for this structure will be released.

⬅️ *Returns*

- `NEONRVM_SUCCESS`: After successful execution.
- `NEONRVM_INVALID_Px`: When facing erroneous parameters.

## 🐍 Python

A new `Param` instance should be created:

```Python
class Param(alpha_init: float, alpha_max: float, alpha_tol: float,
            beta_init: float, basis_percent_min: float, iter_max: int)
```

➡️ *Parameters*

- ⬇️ `[in] alpha_init`: Initial value for *alpha*. Must be a positive and small
    number.
- ⬇️ `[in] alpha_max`: Basis functions associated with *alpha* value beyond this
    limit will be purged. Must be a positive and big number.
- ⬇️ `[in] alpha_tol`: Training session will end if changes in *alpha* values
    gets lower than this value. Must be a positive and small number.
- ⬇️ `[in] beta_init`: Initial value for *beta*. Must be a positive and small
  value.
- ⬇️ `[in] basis_percent_min`: Training session will end if percentage of useful
    basis functions during current training session gets lower than this value.
    Must be a value in `[0.0, 100.0]` range.
- ⬇️ `[in] iter_max`: Training session will end if training loop iteration count
    goes beyond this value. Must be a positive and non-zero number.

⬅️ *Returns*

- A new `Param` instance.

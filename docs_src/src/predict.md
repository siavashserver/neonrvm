# Making predictions

Now that you have the indices and weights of the useful data in hand, you can
make predictions. In order to make prediction with new input data and unknown
outcomes, you should create a new design matrix like what was discussed in *Step
1*, but this time getting populated with closeness and similarities between new
input data, and useful data which we found their indices previously.

If *bias* was found useful during training process, you need to manually append
a column of `1.0` to your new matrix. The new matrix should have row count equal
to the number of new input data samples, and column count equal to the number of
useful basis functions.

Predictions are simply made by multiplying the result matrix and weights vector.
Output vector contains the prediction outcomes, with a length equal to the
number of new input data samples.

## 🚀 C/C++

You can use the `neonrvm_predict` function to make predictions.

```C
int neonrvm_predict(double* phi, double* mu,
                    size_t sample_count, size_t basis_count, double* y)
```

➡️ *Parameters*

- ⬇️ `[in] phi`: Column major matrix, with row count equivalent to the
    *sample_count*, and column count equivalent to the *basis_count*.
- ⬇️ `[in] mu`: Vector of associated basis function weights, with number of
    elements equal to the *basis_count*.
- ⬇️ `[in] sample_count`: Number of input data samples with unknown outcomes.
- ⬇️ `[in] basis_count`: Number of useful basis functions.
- ⬆️ `[out] y`: Vector of predictions made for input data with unknown outcomes,
    with enough room and number of elements equal to the *sample_count*.

⬅️ *Returns*

- `NEONRVM_SUCCESS`: After successful execution.
- `NEONRVM_INVALID_Px`: When facing erroneous parameters.
- `NEONRVM_MATH_ERROR`: When `NaN` or `∞` numbers show up in the calculations.

## 🐍 Python

Number of `phi` columns and `mu` length should match.

```Python
def predict(phi: np.ndarray, mu: np.ndarray)
```

➡️ *Parameters*

- ⬇️ `[in] phi`: Column major matrix, with row count equivalent to the
    *sample_count*, and column count equivalent to the *basis_count*.
- ⬇️ `[in] mu`: Vector of associated basis function weights, with number of
    elements equal to the *basis_count*.

⬅️ *Returns*

- `y: numpy.ndarray`: Vector of predictions made for input data with unknown
  outcomes, with number of elements equal to the *sample_count*.

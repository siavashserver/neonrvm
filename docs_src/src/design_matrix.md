# Design matrix preparation

Design matrix is a 2D matrix with `m` rows and `m` columns, with `m` being
equivalent to the number of input data samples. There is usually a column
consisting of `1.0` appended to that matrix to account for [bias], which makes
it `m*(m+1)` or `m*n`, with `n` representing the number of [basis functions].

In plain English, basis functions show us how much close and similar input data
are to each other. And a [kernel function] decides how much similar our input
data are. Selection of the kernel function depends on the problem at hand, and
you can even mix multiple kernel functions together.

An [RBF] kernel with suitable parameter usually gives satisfactory results. Our
old buddy scikit-learn is there again to help you with a wide selection of
kernel functions and optimizing their parameters.

Before passing your design matrix to the neonrvm, make sure that it's stored in
[column major] order in memory. neonrvm will automatically append an extra
column for bias to the design matrix during *training* process, so you just need
to prepare a 2D `m*m` matrix.

[bias]: https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff
[basis functions]: https://en.wikipedia.org/wiki/Basis_function
[kernel function]: https://en.wikipedia.org/wiki/Kernel_method
[RBF]: https://en.wikipedia.org/wiki/Radial_basis_function
[column major]: https://en.wikipedia.org/wiki/Row-_and_column-major_order

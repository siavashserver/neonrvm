<p align="center">
<img src="https://siavashserver.github.io/neonrvm/neonrvm.svg" alt="neonrvm_logo" title="neonrvm">
</p>

# Introduction

**neonrvm** is an open source machine learning library for performing regression
tasks using [RVM] technique. It is written in C programming language and comes
with bindings for the Python programming language.

neonrvm was born during my master's thesis to help reduce training times and
required system resources. neonrvm did that by getting rid of multiple
middleware layers and optimizing memory usage.

Under the hood neonrvm uses expectation maximization fitting method, and allows
basis functions to be fed incrementally to the model. This helps to keep
training times and memory requirements significantly lower for large data sets.

neonrvm is not trying to be a full featured machine learning framework, and only
provides core training and prediction facilities. You might want to use it in
conjunction with higher level scientific programming languages and machine
learning tool kits instead.

RVM technique is very sensitive to input data representation and kernel
selection. You might consider something else if you are looking for a less
challenging solution.

[RVM]: https://en.wikipedia.org/wiki/Relevance_vector_machine

---

# Documentation

Please visit the dedicated users guide page:
[https://siavashserver.github.io/neonrvm/](https://siavashserver.github.io/neonrvm/)

---

# License

- neonrvm is licensed under the [MIT] license. Please see `LICENSE` for more
    details.

- neonrvm includes code from [Netlib LAPACK] library, which is licensed under a
    [modified BSD license].

- The relevance vector machine is [patented] in the United States by
  [Microsoft].

[MIT]: https://en.wikipedia.org/wiki/MIT_License
[Netlib LAPACK]: http://www.netlib.org/lapack/
[modified BSD license]: http://www.netlib.org/lapack/LICENSE.txt
[patented]: https://patents.google.com/patent/US6633857
[Microsoft]: https://www.microsoft.com

---

# Future work

- Investigate methods to make learning process numerically more stable
- Implement classification
- Create higher level wrappers and programming language bindings
- Improve documentation

---

# Reference

- Tipping, M. E. (2001). Sparse Bayesian learning and the relevance vector machine. Journal of machine learning research, 1(Jun), 211-244.
- Ben-Shimon, D., & Shmilovici, A. (2006). Accelerating the relevance vector machine via data partitioning. Foundations of Computing and Decision Sciences, 31(1), 27-42.

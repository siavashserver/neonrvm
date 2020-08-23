# Using neonrvm

Congratulations, you survived the build process! Following are general tips and
steps in order to train your model and perform predictions using neonrvm.

At this point it's a good idea to grab the original RVM paper and other related
papers to get a feeling of inner workings of the RVM technique and different
parameters. Please have a look at `example.c` and `example.py` for working
sample codes.

In order to keep repetitions in this document lower, Python bindings are briefly
documented. Errors reported by the library, will be raised as exceptions in
Python.

[Sparse Bayesian Models (and the RVM)](http://www.miketipping.com/sparsebayes.htm)

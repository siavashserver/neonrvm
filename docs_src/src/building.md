# Building neonrvm

You can build neonrvm as a dynamic or static library; or manually include
`neonrvm.h` and `neonrvm.c` in your C/C++ project and handle linkage of the
required dependencies.

A C99 compiler is required in order to compile neonrvm, you can find one in
every house these days. You also need [CMake] to configure and generate build
files.

neonrvm requires a linear algebra library providing BLAS/LAPACK interface to do
its magic. Popular ones are [Intel MKL], [OpenBLAS], and the reference [Netlib
LAPACK]. OpenBLAS is almost as fast as Intel MKL, and unlike competition it
doesn't require you to go through a lengthy registration process.

## C library

Please run the following commands inside the source directory to build the
library and examples:

```Shell
$ git clone https://github.com/siavashserver/neonrvm.git
$ cd neonrvm
$ mkdir build
$ cd build
$ cmake ..
$ cmake --build . --config Release
```

It is recommended to use [CPack] (bundled with CMake) to create a nice installer
for your preferred platform. For example to build a `.deb` package, you should
run:

```Shell
$ cpack -G DEB
```

## Python bindings

Python bindings can be installed from the source package using [Flit] Python
module by simply running:

```Shell
$ flit install
```

or from [PyPI] software repository using following command:

```Shell
$ pip install neonrvm
```

[CMake]: https://cmake.org/
[Intel MKL]: https://software.intel.com/mkl
[OpenBLAS]: http://www.openblas.net/
[Netlib LAPACK]: http://www.netlib.org/lapack/
[CPack]: https://cmake.org/cmake/help/latest/module/CPack.html#cpack
[Flit]: https://github.com/takluyver/flit
[PyPI]: https://pypi.python.org/pypi

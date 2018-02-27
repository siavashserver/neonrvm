"""Python bindings for the neonrvm machine learning library."""
__version__ = "0.1.0"

import ctypes as ct
import ctypes.util as ctut

import numpy as np
import numpy.ctypeslib as npct

################################################################################
# Finding and loading the neonrvm library
################################################################################

_path = ctut.find_library("neonrvm")

if _path is None:
    raise RuntimeError("Failed to find neonrvm library,")

_lib = ct.CDLL(_path)


################################################################################
# Dummy neonrvm cache and param data structures
################################################################################

class _cache(ct.Structure):
    pass


class _param(ct.Structure):
    pass


################################################################################
# neonrvm function bindings
################################################################################

_lib.neonrvm_create_cache.restype = ct.c_int
_lib.neonrvm_create_cache.argtypes = [ct.POINTER(ct.POINTER(_cache)),
                                      npct.ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
                                      ct.c_size_t]

_lib.neonrvm_destroy_cache.restype = ct.c_int
_lib.neonrvm_destroy_cache.argtypes = [ct.POINTER(_cache)]

_lib.neonrvm_create_param.restype = ct.c_int
_lib.neonrvm_create_param.argtypes = [ct.POINTER(ct.POINTER(_param)),
                                      ct.c_double,
                                      ct.c_double,
                                      ct.c_double,
                                      ct.c_double,
                                      ct.c_double,
                                      ct.c_size_t]

_lib.neonrvm_destroy_param.restype = ct.c_int
_lib.neonrvm_destroy_param.argtypes = [ct.POINTER(_param)]

_lib.neonrvm_train.restype = ct.c_int
_lib.neonrvm_train.argtypes = [ct.POINTER(_cache),
                               ct.POINTER(_param),
                               ct.POINTER(_param),
                               npct.ndpointer(ct.c_double, flags="F_CONTIGUOUS"),
                               npct.ndpointer(ct.c_size_t, flags="C_CONTIGUOUS"),
                               ct.c_size_t,
                               ct.c_size_t]

_lib.neonrvm_get_training_stats.restype = ct.c_int
_lib.neonrvm_get_training_stats.argtypes = [ct.POINTER(_cache),
                                            ct.POINTER(ct.c_size_t),
                                            ct.POINTER(ct.c_bool)]

_lib.neonrvm_get_training_results.restype = ct.c_int
_lib.neonrvm_get_training_results.argtypes = [ct.POINTER(_cache),
                                              npct.ndpointer(ct.c_size_t, flags="C_CONTIGUOUS"),
                                              npct.ndpointer(ct.c_double, flags="C_CONTIGUOUS")]

_lib.neonrvm_predict.restype = ct.c_int
_lib.neonrvm_predict.argtypes = [npct.ndpointer(ct.c_double, flags="F_CONTIGUOUS"),
                                 npct.ndpointer(ct.c_double, flags="C_CONTIGUOUS"),
                                 ct.c_size_t,
                                 ct.c_size_t,
                                 npct.ndpointer(ct.c_double, flags="C_CONTIGUOUS")]

_lib.neonrvm_get_version.restype = ct.c_int
_lib.neonrvm_get_version.argtypes = [ct.POINTER(ct.c_int),
                                     ct.POINTER(ct.c_int),
                                     ct.POINTER(ct.c_int)]


################################################################################
# Utility functions for internal use
################################################################################

def _gen_err_str(err_code):
    err_str = "neonrvm C API failed with error code: {} - Description: ".format(hex(err_code))

    if err_code & 0x10:
        err_str += "Input parameter #{} passed to the C API is invalid.".format(err_code & 0x0F)
    elif err_code & 0x21:
        err_str += "LAPACKE failed to factorize or solve equations."
    elif err_code & 0x31:
        err_str += "There is NaN or ∞ numbers in calculations."
    else:
        err_str += "Unknown error code has been returned by the C API."

    return err_str


def _get_index_dtype():
    c_size_t_size = ct.sizeof(ct.c_size_t)

    if c_size_t_size is 4:
        return np.uint32
    elif c_size_t_size is 8:
        return np.uint64
    else:
        raise RuntimeError("Unhandled size_t size has been detected.")


################################################################################
# Higher level neonrvm wrappers
################################################################################

class Cache:
    def __init__(self, y: np.ndarray):
        self.c = ct.POINTER(_cache)()

        y = np.ascontiguousarray(y, dtype=np.double)
        count = y.size

        status = _lib.neonrvm_create_cache(ct.byref(self.c), y, count)

        if status is not 0:
            raise RuntimeError(_gen_err_str(status))

    def __del__(self):
        status = _lib.neonrvm_destroy_cache(self.c)

        if status is not 0:
            raise RuntimeError(_gen_err_str(status))


class Param:
    def __init__(self, alpha_init: float, alpha_max: float, alpha_tol: float,
                 beta_init: float, basis_percent_min: float, iter_max: int):
        self.p = ct.POINTER(_param)()

        status = _lib.neonrvm_create_param(ct.byref(self.p), alpha_init, alpha_max, alpha_tol,
                                           beta_init, basis_percent_min, iter_max)

        if status is not 0:
            raise RuntimeError(_gen_err_str(status))

    def __del__(self):
        status = _lib.neonrvm_destroy_param(self.p)

        if status is not 0:
            raise RuntimeError(_gen_err_str(status))


def train(cache: Cache, param1: Param, param2: Param,
          phi: np.ndarray, index: np.ndarray, batch_size_max: int):
    phi = np.asfortranarray(phi, dtype=np.double)
    index = np.ascontiguousarray(index, dtype=_get_index_dtype())
    count = index.size

    status = _lib.neonrvm_train(cache.c, param1.p, param2.p, phi, index, count, batch_size_max)

    if status is not 0:
        raise RuntimeError(_gen_err_str(status))


def get_training_results(cache: Cache):
    basis_count = ct.c_size_t(0)
    bias_used = ct.c_bool(False)

    status = _lib.neonrvm_get_training_stats(cache.c, ct.byref(basis_count), ct.byref(bias_used))

    if status is not 0:
        raise RuntimeError(_gen_err_str(status))

    basis_count = basis_count.value
    bias_used = bias_used.value

    index = np.empty(basis_count, dtype=_get_index_dtype())
    mu = np.empty(basis_count, dtype=np.double)

    index = np.ascontiguousarray(index, dtype=_get_index_dtype())
    mu = np.ascontiguousarray(mu, dtype=np.double)

    status = _lib.neonrvm_get_training_results(cache.c, index, mu)

    if status is not 0:
        raise RuntimeError(_gen_err_str(status))

    return index, mu, basis_count, bias_used


def predict(phi: np.ndarray, mu: np.ndarray):
    phi = np.asfortranarray(phi, dtype=np.double)
    mu = np.ascontiguousarray(mu, dtype=np.double)

    if phi.ndim is 1:
        sample_count = 1
        basis_count = phi.shape[0]
    elif phi.ndim is 2:
        sample_count = phi.shape[0]
        basis_count = phi.shape[1]
    else:
        raise RuntimeError("Unsupported matrix dimension has been encountered.")

    if mu.shape[0] is not basis_count:
        raise RuntimeError("Number of basis functions and weights don't match.")

    y = np.empty(sample_count, dtype=np.double)
    y = np.ascontiguousarray(y, dtype=np.double)

    status = _lib.neonrvm_predict(phi, mu, sample_count, basis_count, y)

    if status is not 0:
        raise RuntimeError(_gen_err_str(status))

    return y


def get_version():
    major = ct.c_int(0)
    minor = ct.c_int(0)
    patch = ct.c_int(0)

    status = _lib.neonrvm_get_version(ct.byref(major), ct.byref(minor), ct.byref(patch))

    if status is not 0:
        raise RuntimeError(_gen_err_str(status))

    return major.value, minor.value, patch.value

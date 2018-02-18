#ifndef NEONRVM_H
#define NEONRVM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>

#if defined(_WIN32)
#if defined(neonrvm_EXPORTS)
#define NEONRVM_API __declspec(dllexport)
#else
#define NEONRVM_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) && defined(neonrvm_EXPORTS)
#define NEONRVM_API __attribute__((visibility("default")))
#else
#define NEONRVM_API
#endif

#define NEONRVM_SUCCESS 0x00
#define NEONRVM_INVALID_P1 0x11
#define NEONRVM_INVALID_P2 0x12
#define NEONRVM_INVALID_P3 0x13
#define NEONRVM_INVALID_P4 0x14
#define NEONRVM_INVALID_P5 0x15
#define NEONRVM_INVALID_P6 0x16
#define NEONRVM_INVALID_P7 0x17
#define NEONRVM_LAPACKE_ERROR 0x21
#define NEONRVM_MATH_ERROR 0x31

typedef struct neonrvm_cache neonrvm_cache;
typedef struct neonrvm_param neonrvm_param;

NEONRVM_API int neonrvm_create_cache(neonrvm_cache** cache, double* y, size_t count);
NEONRVM_API int neonrvm_destroy_cache(neonrvm_cache* cache);

NEONRVM_API int neonrvm_create_param(neonrvm_param** param, double alpha_init, double alpha_max, double alpha_tol, double beta_init, double basis_percent_min, size_t iter_max);
NEONRVM_API int neonrvm_destroy_param(neonrvm_param* param);

NEONRVM_API int neonrvm_train(neonrvm_cache* cache, neonrvm_param* param1, neonrvm_param* param2, double* phi, size_t* index, size_t count, size_t batch_size_max);

NEONRVM_API int neonrvm_get_training_stats(neonrvm_cache* cache, size_t* basis_count, bool* bias_used);
NEONRVM_API int neonrvm_get_training_results(neonrvm_cache* cache, size_t* index, double* mu);

NEONRVM_API int neonrvm_predict(double* phi, double* mu, size_t sample_count, size_t basis_count, double* y);

NEONRVM_API int neonrvm_get_version(int* major, int* minor, int* patch);

#ifdef __cplusplus
}
#endif /* extern "C" */

#endif /* NEONRVM_H */

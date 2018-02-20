#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "neonrvm.h"

/******************************************************************************
* Declaring required CBLAS/LAPACKE functions here so we don't have to deal with
* different header locations and names. (Taken from Netlib's CBLAS/LAPACKE 3.8)
******************************************************************************/

/* CBLAS */
typedef enum { CblasRowMajor = 101,
    CblasColMajor = 102 } CBLAS_LAYOUT;
typedef enum { CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113 } CBLAS_TRANSPOSE;

extern double cblas_ddot(const int N, const double* X, const int incX, const double* Y, const int incY);
extern void cblas_dscal(const int N, const double alpha, double* X, const int incX);
extern void cblas_dgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, const int M, const int N, const double alpha, const double* A, const int lda, const double* X, const int incX, const double beta, double* Y, const int incY);
extern void cblas_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const double alpha, const double* A, const int lda, const double* B, const int ldb, const double beta, double* C, const int ldc);

/* LAPACKE */
#define lapack_int int
#define LAPACK_ROW_MAJOR 101
#define LAPACK_COL_MAJOR 102

extern lapack_int LAPACKE_dpotrf(int matrix_layout, char uplo, lapack_int n, double* a, lapack_int lda);
extern lapack_int LAPACKE_dpotrs(int matrix_layout, char uplo, lapack_int n, lapack_int nrhs, const double* a, lapack_int lda, double* b, lapack_int ldb);

/******************************************************************************
* neonrvm specific macro definitions
******************************************************************************/

#define NEONRVM_VERSION_MAJOR 0
#define NEONRVM_VERSION_MINOR 1
#define NEONRVM_VERSION_PATCH 0

/* Defining private functions as static helps keeping library size low. */
#define NEONRVM_STATIC static

/* This helps to identify uninitialized structures. */
#define NEONRVM_STRUCT_MAGIC 0xACDC

/* Special identifier for bias */
#define NEONRVM_BIAS_MAGIC SIZE_MAX

/* Limiting the matrix dimension to the maximum size allowable by CBLAS/LAPACKE interface  */
#define NEONRVM_MAT_DIM_MAX INT32_MAX

/******************************************************************************
* Utility functions
******************************************************************************/

NEONRVM_STATIC void* nextend(void* mem_addr, size_t count, size_t size, bool keep_content)
{
    if ((0 == count) || (0 == size) || (count > SIZE_MAX / size)) {
        assert(0);
        abort();
    }

    void* addr = NULL;

    if ((true == keep_content) && (NULL != mem_addr)) {
        addr = realloc(mem_addr, count * size);
    } else {
        if (NULL != mem_addr) {
            free(mem_addr);
        }

        addr = malloc(count * size);
    }

    if (NULL == addr) {
        assert(0);
        abort();
    }

    return addr;
}

NEONRVM_STATIC void* nmalloc(size_t count, size_t size)
{
    return nextend(NULL, count, size, false);
}

NEONRVM_STATIC void nfree(void* mem_addr)
{
    if (NULL == mem_addr) {
        assert(0);
        abort();
    }

    free(mem_addr);
}

NEONRVM_STATIC bool is_finite(double* numbers, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        if (0 == isfinite(numbers[i])) {
            return false;
        }
    }

    return true;
}

/******************************************************************************
* Training cache related data types and functions
******************************************************************************/

struct neonrvm_cache {
    size_t struct_initialized;

    double beta;

    /* data prefixed with m_ denote 2D matrices */
    double* m_alpha_diag;
    double* m_phi;
    double* m_phiTphi;
    double* m_sigma;

    /* data prefixed with v_ denote 1D vectors */
    double* v_alpha;
    double* v_alpha_old;
    double* v_error;
    double* v_gamma;
    double* v_mu;
    double* v_phiTy;
    double* v_y;

    size_t* v_index;

    size_t m; /* total sample count */
    size_t n; /* current basis function count */
    size_t n_reserved; /* reserved basis function space count */
    size_t n_train_current; /* number of basis functions used at the beginning of current training session */

    bool bias_used; /* keeping track of bias presence among basis functions during training sessions */
};

NEONRVM_API int neonrvm_create_cache(neonrvm_cache** cache, double* y, size_t count)
{
    if (NULL == cache) {
        assert(0);
        return NEONRVM_INVALID_P1;
    }

    if ((NULL == y) || (false == is_finite(y, count))) {
        assert(0);
        return NEONRVM_INVALID_P2;
    }

    if ((2 > count) || (NEONRVM_MAT_DIM_MAX - 1 < count) || (SIZE_MAX / sizeof(double) < count)) {
        /* Reserved for bias ---------------- ^ */
        assert(0);
        return NEONRVM_INVALID_P3;
    }

    (*cache) = nmalloc(1, sizeof(neonrvm_cache));

    (*cache)->beta = 0.0;

    (*cache)->m_alpha_diag = NULL;
    (*cache)->m_phi = NULL;
    (*cache)->m_phiTphi = NULL;
    (*cache)->m_sigma = NULL;

    (*cache)->v_alpha = NULL;
    (*cache)->v_alpha_old = NULL;
    (*cache)->v_error = nmalloc(count, sizeof(double));
    (*cache)->v_gamma = NULL;
    (*cache)->v_mu = NULL;
    (*cache)->v_phiTy = NULL;
    (*cache)->v_y = nmalloc(count, sizeof(double));
    memcpy((*cache)->v_y, y, count * sizeof(double));

    (*cache)->v_index = NULL;

    (*cache)->m = count;
    (*cache)->n = 0;
    (*cache)->n_reserved = 0;
    (*cache)->n_train_current = 0;
    (*cache)->bias_used = false;

    (*cache)->struct_initialized = NEONRVM_STRUCT_MAGIC;

    return NEONRVM_SUCCESS;
}

NEONRVM_API int neonrvm_destroy_cache(neonrvm_cache* cache)
{
    if ((NULL == cache) || (NEONRVM_STRUCT_MAGIC != cache->struct_initialized)) {
        assert(0);
        return NEONRVM_INVALID_P1;
    }

    /* no need to free up unallocated memory space */
    if (0 != cache->n_reserved) {
        nfree(cache->m_alpha_diag);
        nfree(cache->m_phi);
        nfree(cache->m_phiTphi);
        nfree(cache->m_sigma);

        nfree(cache->v_alpha);
        nfree(cache->v_alpha_old);
        nfree(cache->v_gamma);
        nfree(cache->v_mu);
        nfree(cache->v_phiTy);

        nfree(cache->v_index);
    }

    nfree(cache->v_error);
    nfree(cache->v_y);

    nfree(cache);

    return NEONRVM_SUCCESS;
}

/******************************************************************************
* Training parameter related data types and functions
******************************************************************************/

struct neonrvm_param {
    size_t struct_initialized;

    double alpha_init;
    double alpha_max;
    double alpha_tol;
    double beta_init;

    double pcnt_min; /* minimum percentage of basis functions to keep during every training session */
    size_t iter_max;
};

NEONRVM_API int neonrvm_create_param(neonrvm_param** param, double alpha_init, double alpha_max, double alpha_tol, double beta_init, double basis_percent_min, size_t iter_max)
{
    if (NULL == param) {
        assert(0);
        return NEONRVM_INVALID_P1;
    }

    if ((false == is_finite(&alpha_init, 1)) || (0.0 >= alpha_init)) {
        assert(0);
        return NEONRVM_INVALID_P2;
    }

    if ((false == is_finite(&alpha_max, 1)) || (0.0 >= alpha_max) || (alpha_init >= alpha_max)) {
        assert(0);
        return NEONRVM_INVALID_P3;
    }

    if ((false == is_finite(&alpha_tol, 1)) || (0.0 >= alpha_tol)) {
        assert(0);
        return NEONRVM_INVALID_P4;
    }

    if ((false == is_finite(&beta_init, 1)) || (0.0 >= beta_init)) {
        assert(0);
        return NEONRVM_INVALID_P5;
    }

    if ((false == is_finite(&basis_percent_min, 1)) || (0.0 > basis_percent_min) || (100.0 < basis_percent_min)) {
        assert(0);
        return NEONRVM_INVALID_P6;
    }

    if (1 > iter_max) {
        assert(0);
        return NEONRVM_INVALID_P7;
    }

    (*param) = nmalloc(1, sizeof(neonrvm_param));

    (*param)->alpha_init = alpha_init;
    (*param)->alpha_max = alpha_max;
    (*param)->alpha_tol = alpha_tol;
    (*param)->beta_init = beta_init;

    (*param)->pcnt_min = basis_percent_min;
    (*param)->iter_max = iter_max;

    (*param)->struct_initialized = NEONRVM_STRUCT_MAGIC;

    return NEONRVM_SUCCESS;
}

NEONRVM_API int neonrvm_destroy_param(neonrvm_param* param)
{
    if ((NULL == param) || (NEONRVM_STRUCT_MAGIC != param->struct_initialized)) {
        assert(0);
        return NEONRVM_INVALID_P1;
    }

    nfree(param);

    return NEONRVM_SUCCESS;
}

/******************************************************************************
* Training related functions
******************************************************************************/

NEONRVM_STATIC void resize_cache(neonrvm_cache* c)
{
    /* grow reserved memory space for basis functions by 1.5x */
    if (c->n > 2 * (SIZE_MAX / 3)) {
        /* avoid unsigned integer wrapping */
        assert(0);
        abort();
    }
    c->n_reserved = (3 * c->n) / 2;

    /* cap by num samples + (0: bias was useful in previous iteration, 1: need to add bias again) */
    size_t n_max = c->m + (c->bias_used ? 0 : 1);
    c->n_reserved = c->n_reserved < n_max ? c->n_reserved : n_max;

    /* reallocate space */
    if (c->m > SIZE_MAX / c->n_reserved) {
        /* avoid unsigned integer wrapping */
        assert(0);
        abort();
    }
    size_t mat_count = c->n_reserved * c->n_reserved;
    size_t vec_count = c->n_reserved;
    size_t phi_count = c->m * c->n_reserved;

    c->m_alpha_diag = nextend(c->m_alpha_diag, mat_count, sizeof(double), false);
    c->m_phi = nextend(c->m_phi, phi_count, sizeof(double), true); /* keep content */
    c->m_phiTphi = nextend(c->m_phiTphi, mat_count, sizeof(double), false);
    c->m_sigma = nextend(c->m_sigma, mat_count, sizeof(double), false);

    c->v_alpha = nextend(c->v_alpha, vec_count, sizeof(double), false);
    c->v_alpha_old = nextend(c->v_alpha_old, vec_count, sizeof(double), false);
    c->v_gamma = nextend(c->v_gamma, vec_count, sizeof(double), false);
    c->v_mu = nextend(c->v_mu, vec_count, sizeof(double), false);
    c->v_phiTy = nextend(c->v_phiTy, vec_count, sizeof(double), false);

    c->v_index = nextend(c->v_index, vec_count, sizeof(size_t), true); /* keep content */
}

NEONRVM_STATIC void init_alpha_beta(neonrvm_cache* c, neonrvm_param* p)
{
    for (size_t i = 0; i < c->n; i++) {
        c->v_alpha[i] = p->alpha_init;
        c->v_alpha_old[i] = p->alpha_init;
    }

    c->beta = p->beta_init;
}

NEONRVM_STATIC void add_bias(neonrvm_cache* c)
{
    size_t index_offset = c->m * (c->n - 1);

    /* append a column of 1.0s to the design matrix (phi) */
    for (size_t i = 0; i < c->m; i++) {
        c->m_phi[index_offset + i] = 1.0;
    }

    /* set last index to bias magic number */
    c->v_index[c->n - 1] = NEONRVM_BIAS_MAGIC;
}

NEONRVM_STATIC void update_cache(neonrvm_cache* c, neonrvm_param* p, double* phi, size_t* index, size_t count)
{
    size_t n_old = c->n;

    /* num basis total + (0: bias was useful in previous iteration, 1: need to add bias again) */
    c->n = c->n + count + (c->bias_used ? 0 : 1);
    c->n_train_current = c->n;

    /* make more room if necessary */
    if (c->n > c->n_reserved) {
        resize_cache(c);
    }

    /* (re)initialize alpha and beta */
    init_alpha_beta(c, p);

    /* add new indices and basis functions */
    memcpy(&c->m_phi[c->m * n_old], phi, c->m * count * sizeof(double));
    memcpy(&c->v_index[n_old], index, count * sizeof(size_t));

    /* add bias if necessary */
    if (false == c->bias_used) {
        add_bias(c);
    }

    /* precalculate phi.T*phi and phi.T*y matrices */
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, c->n, c->n, c->m, 1.0, c->m_phi, c->m, c->m_phi, c->m, 0.0, c->m_phiTphi, c->n);
    cblas_dgemv(CblasColMajor, CblasTrans, c->m, c->n, 1.0, c->m_phi, c->m, c->v_y, 1, 0.0, c->v_phiTy, 1);
}

NEONRVM_STATIC bool alpha_max_met(neonrvm_cache* c, neonrvm_param* p)
{
    double diff_max = 0.0;

    for (size_t i = 0; i < c->n; i++) {
        double diff = c->v_alpha[i] - c->v_alpha_old[i];
        diff = diff > 0.0 ? diff : -diff; /* abs */
        diff_max = diff_max > diff ? diff_max : diff; /* max */
    }

    bool result = p->alpha_tol > diff_max;

    return result;
}

NEONRVM_STATIC bool pcnt_min_met(neonrvm_cache* c, neonrvm_param* p)
{
    double pcnt = (c->n * 100.0) / c->n_train_current;

    bool result = pcnt < p->pcnt_min;

    return result;
}

NEONRVM_STATIC bool convergence_conditions_met(neonrvm_cache* c, neonrvm_param* p)
{
    bool condition1 = alpha_max_met(c, p);

    bool condition2 = pcnt_min_met(c, p);

    bool result = condition1 || condition2;

    return result;
}

NEONRVM_STATIC void calc_sigma(neonrvm_cache* c)
{
    memcpy(c->m_sigma, c->m_phiTphi, c->n * c->n * sizeof(double));
    cblas_dscal(c->n * c->n, c->beta, c->m_sigma, 1);

    /* add v_alpha to m_sigma diagonal elements */
    for (size_t i = 0; i < c->n; i++) {
        c->m_sigma[i * c->n + i] += c->v_alpha[i];
    }
}

NEONRVM_STATIC int calc_factor(neonrvm_cache* c)
{
    int status = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', c->n, c->m_sigma, c->n);
    if (0 != status) {
        assert(0);
        return NEONRVM_LAPACKE_ERROR;
    }

    return NEONRVM_SUCCESS;
}

NEONRVM_STATIC int calc_mu(neonrvm_cache* c)
{
    memcpy(c->v_mu, c->v_phiTy, c->n * sizeof(double));
    cblas_dscal(c->n, c->beta, c->v_mu, 1);

    int status = LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'U', c->n, 1, c->m_sigma, c->n, c->v_mu, c->n);
    if (0 != status) {
        assert(0);
        return NEONRVM_LAPACKE_ERROR;
    }

    return NEONRVM_SUCCESS;
}

NEONRVM_STATIC int calc_gamma(neonrvm_cache* c)
{
    /* set diagonal elements to v_alpha */
    memset(c->m_alpha_diag, 0, c->n * c->n * sizeof(double));
    for (size_t i = 0; i < c->n; i++) {
        c->m_alpha_diag[i * c->n + i] = c->v_alpha[i];
    }

    int status = LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'U', c->n, c->n, c->m_sigma, c->n, c->m_alpha_diag, c->n);
    if (0 != status) {
        assert(0);
        return NEONRVM_LAPACKE_ERROR;
    }

    /* extract the diagonal elements and calc final gamma */
    for (size_t i = 0; i < c->n; i++) {
        c->v_gamma[i] = 1.0 - c->m_alpha_diag[i * c->n + i];
    }

    return NEONRVM_SUCCESS;
}

NEONRVM_STATIC void calc_alpha(neonrvm_cache* c)
{
    for (size_t i = 0; i < c->n; i++) {
        c->v_alpha[i] = c->v_gamma[i] / (c->v_mu[i] * c->v_mu[i]);
    }
}

NEONRVM_STATIC void calc_beta(neonrvm_cache* c)
{
    memcpy(c->v_error, c->v_y, c->m * sizeof(double));
    cblas_dgemv(CblasColMajor, CblasNoTrans, c->m, c->n, -1.0, c->m_phi, c->m, c->v_mu, 1, 1.0, c->v_error, 1);

    /* calc sum v_gamma */
    double sum_gamma = 0.0;
    for (size_t i = 0; i < c->n; i++) {
        sum_gamma += c->v_gamma[i];
    }

    c->beta = (c->m - sum_gamma) / cblas_ddot(c->m, c->v_error, 1, c->v_error, 1);
}

NEONRVM_STATIC void filter_basis_funcs(neonrvm_cache* c, neonrvm_param* p)
{
    size_t index_tail = c->n - 1;

    for (size_t i = 0; i < index_tail; i++) {
        if (c->v_alpha[i] > p->alpha_max) {
            for (size_t j = index_tail; j > i; j--) {
                index_tail--;

                if (c->v_alpha[j] < p->alpha_max) {
                    memcpy(&c->m_phi[i * c->m], &c->m_phi[j * c->m], c->m * sizeof(double));

                    memcpy(&c->m_phiTphi[i * c->n], &c->m_phiTphi[j * c->n], c->n * sizeof(double));
                    for (size_t k = 0; k < j; k++) {
                        c->m_phiTphi[k * c->n + i] = c->m_phiTphi[k * c->n + j];
                    }

                    c->v_alpha[i] = c->v_alpha[j];
                    c->v_alpha_old[i] = c->v_alpha_old[j];
                    c->v_mu[i] = c->v_mu[j];
                    c->v_phiTy[i] = c->v_phiTy[j];

                    c->v_index[i] = c->v_index[j];

                    break;
                }
            }
        }
    }

    /* compact phiTphi if there is anything removed */
    if (index_tail != (c->n - 1)) {
        for (size_t i = 0; i < index_tail; i++) {
            memmove(&c->m_phiTphi[(i + 1) * (index_tail + 1)], &c->m_phiTphi[(i + 1) * c->n], (index_tail + 1) * sizeof(double));
        }
    }

    c->n = index_tail + 1;
}

NEONRVM_STATIC int check_numbers(neonrvm_cache* c)
{
    if (false == is_finite(c->v_mu, c->n)) {
        assert(0);
        return NEONRVM_MATH_ERROR;
    }

    if (false == is_finite(c->v_alpha, c->n)) {
        assert(0);
        return NEONRVM_MATH_ERROR;
    }

    if (false == is_finite(&c->beta, 1)) {
        assert(0);
        return NEONRVM_MATH_ERROR;
    }

    return NEONRVM_SUCCESS;
}

NEONRVM_STATIC int main_training_loop(neonrvm_cache* c, neonrvm_param* p)
{
    for (size_t i = 0; i < p->iter_max; i++) {
        int status = NEONRVM_SUCCESS;

        /* check for convergence */
        if ((i > 1) && convergence_conditions_met(c, p)) {
            break;
        }

        memcpy(c->v_alpha_old, c->v_alpha, c->n * sizeof(double));

        calc_sigma(c);

        status = calc_factor(c);
        if (NEONRVM_SUCCESS != status) {
            return status;
        }

        status = calc_mu(c);
        if (NEONRVM_SUCCESS != status) {
            return status;
        }

        status = calc_gamma(c);
        if (NEONRVM_SUCCESS != status) {
            return status;
        }

        calc_alpha(c);

        calc_beta(c);

        filter_basis_funcs(c, p);

        status = check_numbers(c);
        if (NEONRVM_SUCCESS != status) {
            return status;
        }
    }

    return NEONRVM_SUCCESS;
}

NEONRVM_STATIC bool contains_bias(size_t* numbers, size_t count)
{
    for (size_t i = 0; i < count; i++) {
        if (NEONRVM_BIAS_MAGIC == numbers[i]) {
            return true;
        }
    }

    return false;
}

NEONRVM_STATIC int train_incremental(neonrvm_cache* c, neonrvm_param* p, double* phi, size_t* index, size_t count)
{
    update_cache(c, p, phi, index, count);

    int status = main_training_loop(c, p);
    if (NEONRVM_SUCCESS != status) {
        return status;
    }

    c->bias_used = contains_bias(c->v_index, c->n);

    return NEONRVM_SUCCESS;
}

NEONRVM_STATIC void move_bias(neonrvm_cache* c)
{
    size_t index_tail = c->n - 1;

    for (size_t i = 0; i < c->n; i++) {
        if (NEONRVM_BIAS_MAGIC == c->v_index[i]) {
            c->v_index[i] = c->v_index[index_tail];
            c->v_index[index_tail] = NEONRVM_BIAS_MAGIC;

            double mu_bias = c->v_mu[i];
            c->v_mu[i] = c->v_mu[index_tail];
            c->v_mu[index_tail] = mu_bias;

            memcpy(&c->m_phi[i * c->m], &c->m_phi[index_tail * c->m], c->m * sizeof(double));
            add_bias(c);
        }
    }
}

NEONRVM_API int neonrvm_train(neonrvm_cache* cache, neonrvm_param* param1, neonrvm_param* param2, double* phi, size_t* index, size_t count, size_t batch_size_max)
{
    int status = NEONRVM_SUCCESS;

    if ((NULL == cache) || (NEONRVM_STRUCT_MAGIC != cache->struct_initialized)) {
        assert(0);
        return NEONRVM_INVALID_P1;
    }

    if ((NULL == param1) || (NEONRVM_STRUCT_MAGIC != param1->struct_initialized)) {
        assert(0);
        return NEONRVM_INVALID_P2;
    }

    if ((NULL == param2) || (NEONRVM_STRUCT_MAGIC != param2->struct_initialized)) {
        assert(0);
        return NEONRVM_INVALID_P3;
    }

    if ((NULL == phi) || (false == is_finite(phi, cache->m * count))) {
        assert(0);
        return NEONRVM_INVALID_P4;
    }

    if ((NULL == index) || contains_bias(index, count)) {
        assert(0);
        return NEONRVM_INVALID_P5;
    }

    if ((1 > count) || (count > cache->m - cache->n)) {
        assert(0);
        return NEONRVM_INVALID_P6;
    }

    if (1 > batch_size_max) {
        assert(0);
        return NEONRVM_INVALID_P7;
    }

    /* batch_size_max = min(num_basis, batch_size_max) */
    batch_size_max = batch_size_max < count ? batch_size_max : count;

    /* incremental training param: param1 */
    size_t i = 0;
    for (i = 0; i < (count / batch_size_max); i++) {
        double* batch_phi = &phi[i * batch_size_max * cache->m];
        size_t* batch_index = &index[i * batch_size_max];

        status = train_incremental(cache, param1, batch_phi, batch_index, batch_size_max);

        if (NEONRVM_SUCCESS != status) {
            return status;
        }
    }

    /* final polish param: param2 */
    size_t basis_funcs_left = count % batch_size_max;
    double* batch_phi = &phi[i * batch_size_max * cache->m];
    size_t* batch_index = &index[i * batch_size_max];
    status = train_incremental(cache, param2, batch_phi, batch_index, basis_funcs_left);

    if (NEONRVM_SUCCESS != status) {
        return status;
    }

    /* move bias related data to the end of list */
    if (true == cache->bias_used) {
        move_bias(cache);
    }

    return NEONRVM_SUCCESS;
}

/******************************************************************************
* Training results related functions
******************************************************************************/

NEONRVM_API int neonrvm_get_training_stats(neonrvm_cache* cache, size_t* basis_count, bool* bias_used)
{
    if ((NULL == cache) || (NEONRVM_STRUCT_MAGIC != cache->struct_initialized)) {
        assert(0);
        return NEONRVM_INVALID_P1;
    }

    if (NULL == basis_count) {
        assert(0);
        return NEONRVM_INVALID_P2;
    }

    if (NULL == bias_used) {
        assert(0);
        return NEONRVM_INVALID_P3;
    }

    (*basis_count) = cache->n;
    (*bias_used) = cache->bias_used;

    return NEONRVM_SUCCESS;
}

NEONRVM_API int neonrvm_get_training_results(neonrvm_cache* cache, size_t* index, double* mu)
{
    if ((NULL == cache) || (NEONRVM_STRUCT_MAGIC != cache->struct_initialized)) {
        assert(0);
        return NEONRVM_INVALID_P1;
    }

    if (NULL == index) {
        assert(0);
        return NEONRVM_INVALID_P2;
    }

    if (NULL == mu) {
        assert(0);
        return NEONRVM_INVALID_P3;
    }

    memcpy(index, cache->v_index, cache->n * sizeof(size_t));
    memcpy(mu, cache->v_mu, cache->n * sizeof(double));

    return NEONRVM_SUCCESS;
}

/******************************************************************************
* Prediction related functions
******************************************************************************/

NEONRVM_API int neonrvm_predict(double* phi, double* mu, size_t sample_count, size_t basis_count, double* y)
{
    if ((NULL == phi) || (false == is_finite(phi, sample_count * basis_count))) {
        assert(0);
        return NEONRVM_INVALID_P1;
    }

    if ((NULL == mu) || (false == is_finite(mu, basis_count))) {
        assert(0);
        return NEONRVM_INVALID_P2;
    }

    if ((1 > sample_count) || (NEONRVM_MAT_DIM_MAX < sample_count)) {
        assert(0);
        return NEONRVM_INVALID_P3;
    }

    if ((1 > basis_count) || (NEONRVM_MAT_DIM_MAX < basis_count)) {
        assert(0);
        return NEONRVM_INVALID_P4;
    }

    if (NULL == y) {
        assert(0);
        return NEONRVM_INVALID_P5;
    }

    if (1 == sample_count) {
        (*y) = cblas_ddot(basis_count, phi, 1, mu, 1);
    } else {
        cblas_dgemv(CblasColMajor, CblasNoTrans, sample_count, basis_count, 1.0, phi, sample_count, mu, 1, 0.0, y, 1);
    }

    if (false == is_finite(y, sample_count)) {
        assert(0);
        return NEONRVM_MATH_ERROR;
    }

    return NEONRVM_SUCCESS;
}

/******************************************************************************
* Miscellaneous functions
******************************************************************************/

NEONRVM_API int neonrvm_get_version(int* major, int* minor, int* patch)
{
    if (NULL == major) {
        assert(0);
        return NEONRVM_INVALID_P1;
    }

    if (NULL == minor) {
        assert(0);
        return NEONRVM_INVALID_P2;
    }

    if (NULL == patch) {
        assert(0);
        return NEONRVM_INVALID_P3;
    }

    (*major) = NEONRVM_VERSION_MAJOR;
    (*minor) = NEONRVM_VERSION_MINOR;
    (*patch) = NEONRVM_VERSION_PATCH;

    return NEONRVM_SUCCESS;
}

#include <malloc.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "neonrvm.h"

void sinc(size_t num_samples, double range_x, double scale_y, double scale_noise, double* sinc_x, double* sinc_y)
{
    double PI = 3.1415926535897932;

    for (size_t i = 0; i < num_samples; i++) {
        double x = -range_x + i * (2.0 * range_x / num_samples);
        double y = (x == 0.0) ? 1.0 : sin(PI * x) / (PI * x);

        double noise = ((i % 2) == 0) ? 1.0 : -1.0;

        y = scale_y * y + scale_noise * noise;

        sinc_x[i] = x;
        sinc_y[i] = y;
    }
}

void preprocess(size_t num_samples, double* data)
{
    /* normalize data with avg of 0.0, and std of 1.0 */
    double sum = 0.0;
    for (size_t i = 0; i < num_samples; i++) {
        sum += data[i];
    }

    double avg = sum / num_samples;

    sum = 0.0;
    for (size_t i = 0; i < num_samples; i++) {
        double diff = data[i] - avg;
        sum += diff * diff;
    }

    double vrc = sum / num_samples;

    double std = sqrt(vrc);

    for (size_t i = 0; i < num_samples; i++) {
        data[i] = (data[i] - avg) / std;
    }
}

void rbf(size_t num_x1, size_t num_x2, double* x1, double* x2, double gamma, bool add_bias, double* phi)
{
    /* rbf (x1, x2) = exp( -gamma * (x1-x2)^2 ) */
    /* create a matrix with num_x1 rows (samples), and num_x2 columns (relevant vectors) */
    for (size_t i = 0; i < num_x2; i++) {
        for (size_t j = 0; j < num_x1; j++) {
            double distance = x2[i] - x1[j];
            double k = exp(-gamma * distance * distance);

            phi[i * num_x1 + j] = k;
        }
    }

    /* setting last column to 1.0s, assuming phi does have enough space for an extra column */
    if (add_bias) {
        for (size_t i = 0; i < num_x1; i++) {
            phi[num_x1 * num_x2 + i] = 1.0;
        }
    }
}

double calc_mape(size_t num_samples, double* y_true, double* y_predicted)
{
    double sum_abs_pcnt_err = 0.0;

    for (size_t i = 0; i < num_samples; i++) {
        double y_t = y_true[i];
        double y_p = y_predicted[i];
        double pcnt_err = (y_t - y_p) / y_t;
        double abs_pcnt_err = pcnt_err > 0.0 ? pcnt_err : -pcnt_err;
        sum_abs_pcnt_err += abs_pcnt_err;
    }

    double mape = 100.0 * sum_abs_pcnt_err / num_samples;

    return mape;
}

int main()
{
    /* generate input data using sinc function */
    size_t num_samples = 500;
    double range_x = 5.0; /* -range_x ... +range_x */
    double scale_y = 5.0;
    double scale_noise = 0.001;

    double* x = malloc(num_samples * sizeof(double));
    double* y = malloc(num_samples * sizeof(double));
    double* x_pp = malloc(num_samples * sizeof(double));

    sinc(num_samples, range_x, scale_y, scale_noise, x, y);
    memcpy(x_pp, x, num_samples * sizeof(double));

    /* preprocess input data */
    preprocess(num_samples, x_pp);

    /* make ready the design matrix and data indices */
    bool bias_used = false;
    double gamma = 10.0;

    double* design_matrix = malloc(num_samples * num_samples * sizeof(double));

    rbf(num_samples, num_samples, x_pp, x_pp, gamma, bias_used, design_matrix);

    size_t* indices = malloc(num_samples * sizeof(size_t));
    for (size_t i = 0; i < num_samples; i++) {
        indices[i] = i;
    }

    /* setting learning parameters */
    neonrvm_cache* c;
    neonrvm_param *p1, *p2;

    neonrvm_create_cache(&c, y, num_samples);
    neonrvm_create_param(&p1, 1e-6, 1e3, 1e-1, 1e-6, 80.0, 100);
    neonrvm_create_param(&p2, 1e-6, 1e3, 1e-2, 1e-6, 00.0, 300);

    /* start learning process */
    size_t batch_size_max = 100;

    if (NEONRVM_SUCCESS != neonrvm_train(c, p1, p2, design_matrix, indices, num_samples, batch_size_max)) {
        return EXIT_FAILURE;
    }

    /* gather basis function indices and their associated weights */
    size_t basis_count = 0;
    neonrvm_get_training_stats(c, &basis_count, &bias_used);
    size_t basis_count_without_bias = bias_used ? basis_count - 1 : basis_count;

    size_t* indices_relevant = malloc(basis_count * sizeof(size_t));
    double* weights_relevant = malloc(basis_count * sizeof(double));
    neonrvm_get_training_results(c, indices_relevant, weights_relevant);

    /* print out the relevant vectors */
    for (size_t i = 0; i < basis_count_without_bias; i++) {
        size_t index = indices_relevant[i];
        printf("item: %zu, index: %zu, mu: %f, x: %f, y: %f\n", i, index, weights_relevant[i], x[index], y[index]);
    }

    if (true == bias_used) {
        printf("item: bias, mu: %f\n", weights_relevant[basis_count_without_bias]);
    }

    /* test the model performance using the already preprocessed training data for the sake of simplicity */
    double* x_pp_relevant = malloc(basis_count_without_bias * sizeof(double));
    for (size_t i = 0; i < basis_count_without_bias; i++) {
        size_t index = indices_relevant[i];
        x_pp_relevant[i] = x_pp[index];
    }

    double* phi_relevant = malloc(num_samples * basis_count * sizeof(double));
    rbf(num_samples, basis_count_without_bias, x_pp, x_pp_relevant, gamma, bias_used, phi_relevant);

    double* y_predicted = malloc(num_samples * sizeof(double));
    if (NEONRVM_SUCCESS != neonrvm_predict(phi_relevant, weights_relevant, num_samples, basis_count, y_predicted)) {
        return EXIT_FAILURE;
    }

    /* print out prediction and training stats */
    printf("Mean Absolute Percentage Error: %f\n", calc_mape(num_samples, y, y_predicted));
    printf("Percentage of Relevance Vectors: %f\n", 100.0 * basis_count / num_samples);

    /* clean up */
    neonrvm_destroy_cache(c);
    neonrvm_destroy_param(p1);
    neonrvm_destroy_param(p2);

    free(x);
    free(y);
    free(x_pp);
    free(design_matrix);
    free(indices);
    free(indices_relevant);
    free(weights_relevant);
    free(x_pp_relevant);
    free(phi_relevant);
    free(y_predicted);

    return EXIT_SUCCESS;
}

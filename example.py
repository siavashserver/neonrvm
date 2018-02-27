import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

import neonrvm


def generate_input_data(num_samples, range_x, scale_y, scale_noise):
    index = np.arange(num_samples)

    x = np.linspace(-range_x, range_x, num_samples, False).reshape(-1, 1)

    x_pp = StandardScaler().fit(x).transform(x)

    noise_y = np.empty(num_samples).reshape(-1, 1)
    for i in range(num_samples):
        noise_y[i] = 1.0 if (i % 2 == 0) else -1.0

    y = scale_y * np.sinc(x) + scale_noise * noise_y

    return index, x, x_pp, y


def generate_design_matrix(x, y, gamma, bias_used=False):
    design_matrix = rbf_kernel(x, y, gamma)

    if bias_used is True:
        design_matrix = np.append(design_matrix, np.ones_like(x), axis=1)

    return design_matrix


# Generate sample input data using sinc function
index_basis, x, x_pp, y = generate_input_data(500, 5.0, 5.0, 1e-3)

# Preparing the design matrix
gamma = 10.0
design_matrix = generate_design_matrix(x_pp, x_pp, gamma)

# Setting learning parameters
c = neonrvm.Cache(y)
p1 = neonrvm.Param(1e-6, 1e3, 1e-1, 1e-6, 80.0, 100)
p2 = neonrvm.Param(1e-6, 1e3, 1e-2, 1e-6, 00.0, 300)

# Starting the learning process
neonrvm.train(c, p1, p2, design_matrix, index_basis, 100)

# Getting back the training results
index_relevant, mu, basis_count, bias_used = neonrvm.get_training_results(c)

# Printing the useful input data, and their associated weights
if bias_used is True:
    index_relevant = index_relevant[:-1]

x_rel = x[index_relevant]
y_rel = y[index_relevant]

for i in range(index_relevant.size):
    index = index_relevant[i]
    print("item: {:2d}, index: {:3d}, mu: {:6.3f}, x: {:6.3f}, y: {:6.3f}".format(i, index_basis[index], mu[i],
                                                                                  x.item((index, 0)),
                                                                                  y.item((index, 0))))

if bias_used is True:
    print("item: bias, mu: {:6.3f}".format(mu[-1]))

# Test the model performance using the already preprocessed training data for the sake of simplicity
x_pp_rel = x_pp[index_relevant]

phi_rel = generate_design_matrix(x_pp, x_pp_rel, gamma, bias_used)

y_predicted = neonrvm.predict(phi_rel, mu).reshape(-1, 1)

# Printing prediction and training stats
print("Mean Absolute Percentage Error:  {:.2f}".format(100.0 * np.mean(np.abs((y - y_predicted) / y))))
print("Percentage of Relevance Vectors: {:.2f}".format(100.0 * basis_count / x.size))

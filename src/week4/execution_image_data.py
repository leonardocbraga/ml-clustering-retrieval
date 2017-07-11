import numpy as np
import graphlab as gl
from matplotlib import pyplot as plt
from visualization import *
from expectation_maximization import *

images = gl.SFrame('../../data/images.sf')
images['rgb'] = images.pack_columns(['red', 'green', 'blue'])['X4']

np.random.seed(1)

# Initalize parameters
init_means = [images['rgb'][x] for x in np.random.choice(len(images), 4, replace=False)]
cov = np.diag([images['red'].var(), images['green'].var(), images['blue'].var()])
init_covariances = [cov, cov, cov, cov]
init_weights = [1/4., 1/4., 1/4., 1/4.]

# Convert rgb data to numpy arrays
img_data = [np.array(i) for i in images['rgb']]  

# Run our EM algorithm on the image data using the above initializations. 
out = EM(img_data, init_means, init_covariances, init_weights)

# Plot loglikelihoods
loglikelihoods = out['loglik']
visualize_loglikelihoods(loglikelihoods)

# Plot responsibilities
N, K = out['resp'].shape
random_resp = np.random.dirichlet(np.ones(K), N)
plot_responsibilities_in_RB(images, random_resp, 'Random responsibilities')

out = EM(img_data, init_means, init_covariances, init_weights, maxiter=1)
plot_responsibilities_in_RB(images, out['resp'], 'After 1 iteration')

out = EM(img_data, init_means, init_covariances, init_weights, maxiter=20)
plot_responsibilities_in_RB(images, out['resp'], 'After 20 iterations')

#Plot top 5 images
assignments = compute_image_assignments(images, out)

for component_id in range(4):
    imgs = get_top_images(assignments, component_id)
    plot_top_images(imgs)

import argparse
import joblib
import numpy as np
from pyspark import SparkConf, SparkContext
import scipy
import scipy.ndimage
import scipy.sparse as sparse
import sklearn.metrics.pairwise as pairwise

def median_difference_parallel(image, indices):
    """
    Computes the absolute median difference using only joblib.
    """
    flattened = np.ravel(image)
    out = joblib.Parallel(n_jobs = -1, verbose = 10)(
            joblib.delayed(np.abs)(
                flattened[i] - flattened[j]) for i, j in indices)
    return np.array(out)

def median_difference_iterative(image):
    """
    Computes the median absolute gray-level difference between all the
    pixels in the image.

    Parameters
    ----------
    image : array, shape (H, W)
        Matrix of gray-level intensities.

    Returns
    -------
    md : float
        Median absolute gray-level difference between all pixels.
    """
    D = []
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            startrow = np.max([0, i - 1])
            endrow = np.min([image.shape[0], i + 2])
            startcol = np.max([0, j - 1])
            endcol = np.min([image.shape[1], j + 2])
            patch = image[startrow:endrow, startcol:endcol]
            locali = 0 if i - 1 < 0 else 1
            localj = 0 if j - 1 < 0 else 1
            D.extend(_differences(patch, locali, localj))

    return np.median(np.array(D))

def pairwise_affinities(pixels):
    """
    Computes the affinity for a pair of pixels.
    """
    i, j = pixels
    image = IMAGE.value
    sigma = SIGMA.value
    rbf = pairwise.rbf_kernel(image[i], image[j], gamma = sigma)[0, 0]
    return [(i, [j, rbf]), (j, [i, rbf])]

def assemble_row(affinities):
    """
    Assembles the affinities into a correct row vector.
    """
    rowid, values = affinities
    return [rowid, {v[0]: v[1] for v in values}]

def connectivity(nx, ny, neighborhood = 8):
    """
    Comparable to scikit-learn's grid_to_graph method, though this stops short of computing a full graph and instead simply returns indices in an image that are connected.

    Parameters
    ----------
    nx : integer
        Number of rows (image height).
    ny : integer
        Number of columns (image width).

    Returns
    -------
    indices : array, shape (N, 2)
        The connectivity graph for the image.
    """
    n_voxels = nx * ny
    vertices = np.arange(n_voxels).reshape((nx, ny))
    edges_right = np.vstack((vertices[:, :-1].ravel(), vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))

    edges = None
    if neighborhood == 4:
        edges = np.hstack((edges_right, edges_down))
    elif neighborhood == 8:
        edges_tc = np.vstack((vertices[1:, :-1].ravel(), vertices[:-1, 1:].ravel()))
        edges_bc = np.vstack((vertices[:-1, :-1].ravel(), vertices[1:, 1:].ravel()))
        edges = np.hstack((edges_right, edges_down, edges_tc, edges_bc))
    else:
        quit('ERROR: Unrecognized neighborhood size %s.' % neighborhood)

    return edges.T

def legacy(nx, ny, neighborhood = 8):
    n_voxels = nx * ny
    vertices = np.arange(n_voxels).reshape((nx, ny))
    edges_right = np.vstack((vertices[:, :-1].ravel(), vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))

    edges = None
    if neighborhood == 4:
        edges = np.hstack((edges_right, edges_down))
    elif neighborhood == 8:
        edges_tc = np.vstack((vertices[1:, :-1].ravel(), vertices[:-1, 1:].ravel()))
        edges_bc = np.vstack((vertices[:-1, :-1].ravel(), vertices[1:, 1:].ravel()))
        edges = np.hstack((edges_right, edges_down, edges_tc, edges_bc))
    else:
        quit('ERROR: Unrecognized neighborhood size %s.' % neighborhood)

    weights = np.ones(edges.shape[1])
    diag = np.ones(n_voxels)
    diag_idx = np.arange(n_voxels)
    i_idx = np.hstack((edges[0], edges[1]))
    j_idx = np.hstack((edges[1], edges[0]))

    graph = sparse.coo_matrix(
        (np.hstack((weights, weights, diag)), (np.hstack((i_idx, diag_idx)),
        np.hstack((j_idx, diag_idx)))),
        shape = (n_voxels, n_voxels),
        dtype = np.int)
    return graph

def _differences(patch, i, j):
    """
    Computes the absolute gray-level difference between all the
    pixels in the patch, relative to the current position. If the patch is
    not 3x3, i and j are used to determine where the current pixel is.

    Parameters
    ----------
    patch : array, shape (N, M)
        Image patch. N * M is always either 4, 6, or 8.
    i : integer
        Row of the current pixel (ignored if N == M == 3).
    j : integer
        Column of the current pixel (ignored if N == M == 3).

    Returns
    -------
    diffs : array, shape (P,)
        List of gray-level absolute differences.
    """
    if patch.shape[0] == 3: i = 1
    if patch.shape[1] == 3: j = 1
    differences = []
    for a in range(0, patch.shape[0]):
        for b in range(0, patch.shape[1]):
            if a == i and b == j: continue
            differences.append(np.abs(patch[a, b] - patch[i, j]))
    return np.array(differences)

def image_affinities(image, q = 1.5, gamma = 0.0):
    """
    Calculates a sparse affinity matrix from image data, where each pixel is
    connected only to its (at most) 8 neighbors. Furthermore, the sigma used
    is computed on a local basis.

    Parameters
    ----------
    image : array, shape (P, Q)
        Grayscale image.
    q : float
        Multiplier to compute gamma.
    gamma : float
        If specified and positive, this overrides the use of the multiplier q
        and of computing gamma on a per-neighborhood basis.

    Returns
    -------
    A : array, shape (P * Q, P * Q)
        Symmetric affinity matrix.
    """
    std = gamma
    if gamma <= 0.0:
        med = median_difference_iterative(image)
        std = 1.0 / (2 * ((med * q) ** 2))
    graph = legacy(image.shape[1], image.shape[0])
    connections = graph.nonzero()
    A = sparse.lil_matrix(graph.shape)

    # For each non-zero connection, compute the affinity.
    # We have to do this one at a time in a loop; rbf_kernel() doesn't have
    # a sparse mode, and therefore computing all the affinities at once--even
    # sparse ones--could overwhelm system memory.
    for i, j in zip(connections[0], connections[1]):
        if A[i, j] > 0.0: continue

        # Where do the pixels reside?
        r1 = i / image.shape[1]
        c1 = i % image.shape[1]
        r2 = j / image.shape[1]
        c2 = j % image.shape[1]

        # Compute the RBF value.
        rbf = pairwise.rbf_kernel(image[r1, c1], image[r2, c2], gamma = std)[0, 0]
        A[i, j] = rbf
        A[j, i] = rbf
        A[i, i] = 1.0
        A[j, j] = 1.0

    #return A
    return np.array(A.todense())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'PySpark Affinities',
        epilog = 'lol sp3c+r4l', add_help = 'How to use',
        prog = 'python affinities.py <arguments>')

    # Required arguments.
    parser.add_argument("-i", "--input", required = True,
        help = "Path to an image. Any image!")
    #parser.add_argument("-o", "--output", required = True,
    #    help = "Path to an output directory. If it doesn't exist, it is created.")

    # Optional arguments.
    parser.add_argument("-s", "--sigma", type = float, default = 0.0,
        help = "If specified, the value of sigma to use in computing affinities. [DEFAULT: 0.0]")
    parser.add_argument("-q", "--multiplier", type = float, default = 1.5,
        help = "Multiplier for sigma. [DEFAULT: 1.5]")

    args = vars(parser.parse_args())

    # Set up the Spark context. Because awesome.
    sc = SparkContext(conf = SparkConf())

    # Read in the image. Broadcast it and determine the indices of connected pixels.
    image = scipy.ndimage.imread(args['input'], flatten = True)
    IMAGE = sc.broadcast(np.ravel(image))
    A = sc.parallelize(connectivity(image.shape[0], image.shape[1]), sc.defaultParallelism * 4)

    # If sigma was not specified, we'll compute it ourselves. We do this by first
    # finding the *median absolute gray-level intensity difference* between all
    # connected pixels, and use that to compute sigma.
    sigma = args['sigma']
    if sigma <= 0.0:
        d = np.median(np.array(
                A.map(
                    lambda x: np.abs(IMAGE.value[x[0]] - IMAGE.value[x[1]])
                     )
                .collect()))
        sigma = 1.0 / (2 * ((d * args['multiplier']) ** 2))

    # Now that we have sigma, let's compute an affinity matrix.
    SIGMA = sc.broadcast(sigma)
    affinities = A.flatMap(pairwise_affinities).groupByKey().map(assemble_row).sortByKey()

    d = affinities.collect()
    num = image.shape[0] * image.shape[1]
    A1 = sparse.dok_matrix((num, num), dtype = np.float) 
    for rowid, values in d:
        for k, v in values.iteritems():
            A1[rowid, k] = v
    diag = np.arange(num, dtype = np.int)
    A1[diag, diag] = 1.0
    #A2 = image_affinities(image)
    #np.testing.assert_array_equal(A1, A2)
    print A1.shape
    print A1


import argparse
import numpy as np
import numpy.linalg as la
from pyspark import SparkConf, SparkContext
import scipy
import scipy.ndimage
import scipy.sparse as sparse
import sklearn.metrics.pairwise as pairwise

from image_affinities import connectivity

def estimate_sigma(x):
    if TYPE.value == 'image':
        return np.abs(DATA.value[x[0]] - DATA.value[x[1]])
    else:  # type is text
        x1 = x[0]
        x2 = x[1]
        _, data1 = int(x1[0]), x1[1]
        _, data2 = int(x2[0]), x2[1]

        # Now parse out the floating point coordinates.
        x = np.array(map(float, data1.strip().split(",")))
        y = np.array(map(float, data2.strip().split(",")))
        return la.norm(x - y)

def pairwise_pixels(pixels):
    """
    Computes the affinity for a pair of pixels.
    """
    i, j = pixels
    image = DATA.value
    sigma = SIGMA.value
    rbf = pairwise.rbf_kernel(image[i], image[j], gamma = sigma)[0, 0]
    return [(i, [j, rbf]), (j, [i, rbf])]

def pixel_row_vector(affinities):
    """
    Assembles the affinities into a correct row vector.
    """
    rowid, values = affinities
    return [rowid, {v[0]: v[1] for v in values}]

def pairwise_points(x):
    """
    Computes the RBF affinity for a pair of points.
    """
    x1 = x[0]
    x2 = x[1]
    i1, data1 = int(x1[0]), x1[1]
    i2, data2 = int(x2[0]), x2[1]

    # Now parse out the floating point coordinates.
    x = np.array(map(float, data1.strip().split(",")))
    y = np.array(map(float, data2.strip().split(",")))

    # Find the RBF kernel between them, assuming their distance is within
    # the threshold.
    epsilon = DATA.value
    sigma = SIGMA.value

    # Are these two points close enough together?
    threshold = la.norm(x - y) < epsilon if epsilon > 0.0 else True
    rbf = pairwise.rbf_kernel(x, y, gamma = sigma)[0, 0] if threshold else 0.0
    return [i1, [i2, rbf]]

    # We don't need to return a pair of tuples, because all pairings
    # are enumerated; there will be another case where the values of
    # i1 and i2 are switched.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'PySpark Affinities',
        epilog = 'lol sp3c+r4l', add_help = 'How to use',
        prog = 'python compute_affinities.py [image | text] <arguments>')
    parser.add_argument("-i", "--input", required = True,
        help = "Path to an image file, or text file with n-dimensional comma-separated Cartesian data.")
    parser.add_argument("-o", "--output", required = True,
        help = "Output path.")

    parser.add_argument("-s", "--sigma", type = float, default = 0.0,
        help = "If > 0.0, the value of sigma used in RBF kernel. Otherwise, it is estimated. [DEFAULT: 0.0]")
    parser.add_argument("-q", "--multiplier", type = float, default = 1.0,
        help = "Constant multiplier for sigma in the RBF kernel. [DEFAULT: 1.0]")

    # Create subparsers for images and text.
    subparsers = parser.add_subparsers(dest = "sub_name")

    text_p = subparsers.add_parser("text")
    text_p.add_argument("-e", "--epsilon", type = float, default = 2.0,
        help = "If specified, pairwise affinities are only computed for points whose Euclidean distance is less than this. [DEFAULT: 2.0]")

    image_p = subparsers.add_parser("image")
    image_p.add_argument("-n", "--neighborhood", type = int, choices = [4, 8], default = 8,
        help = "Number of connected pixels in the neighborhood, 4 or 8. [DEFAULT: 8]")

    args = vars(parser.parse_args())
    infile = args['input']
    outdir = args['output']
    sigma = args['sigma']
    q = args['multiplier']

    # Set up the Spark context. Because awesome.
    sc = SparkContext(conf = SparkConf())
    if args['sub_name'] != 'image' and args['sub_name'] != 'text':
        print 'Command "%s" not recognized.' % args['sub_name']
        quit()

    TYPE = sc.broadcast(args['sub_name'])
    A = None
    if args["sub_name"] == "image":
        # Read in the image. Broadcast it and determine the indices of connected pixels.
        image = scipy.ndimage.imread(args['input'], flatten = True)
        DATA = sc.broadcast(image.ravel())
        A = sc.parallelize(connectivity(image.shape[0], image.shape[1]), sc.defaultParallelism * 4)
    else:
        # Read the input file, index each data point, and parallelize to an RDD.
        rawdata = np.loadtxt(args['input'], dtype = np.str, delimiter = "\n")
        indexed = np.vstack([np.arange(rawdata.shape[0]), rawdata]).T
        DATA = sc.broadcast(args['epsilon'])
        D = sc.parallelize(indexed)
        A = D.cartesian(D)

    # If sigma was not specified, we'll compute it ourselves. We do this by first
    # finding the *median absolute gray-level intensity difference* between all
    # connected pixels, and use that to compute sigma.
    if sigma <= 0.0:
        d = np.median(np.array(
                A.map(estimate_sigma)
                .collect()))
        sigma = 1.0 / (2 * ((d * q) ** 2))

    # Now that we have sigma, let's compute an affinity matrix.
    SIGMA = sc.broadcast(sigma)

    if args['sub_name'] == 'image':
        affinities = A.flatMap(pairwise_pixels).groupByKey().map(pixel_row_vector).sortByKey().collect()

        num = image.shape[0] * image.shape[1]
        A1 = sparse.dok_matrix((num, num), dtype = np.float)
        for rowid, values in affinities:
            for k, v in values.iteritems():
                A1[rowid, k] = v
        diag = np.arange(num, dtype = np.int)
        A1[diag, diag] = 1.0
        #A2 = image_affinities(image)
        #np.testing.assert_array_equal(A1, A2)
        print A1.shape
        print A1
    else:
        affinities = A.map(pairwise_points).filter(lambda x: x[1][1] > 0.0).sortByKey().collect()

        print 'Of %sx%s possible pairs, we have %s.' % (rawdata.shape[0], rawdata.shape[0], len(affinities))

import argparse
import numpy as np
import numpy.linalg as la
from pyspark import SparkConf, SparkContext
import scipy
import scipy.ndimage
import scipy.sparse as sparse
import sklearn.metrics.pairwise as pairwise

from image_affinities import connectivity, parse_coordinates

def distance_threshold(z):
    """
    Firstly, this method is only invoked if the distance threshold epsilon is set.
    Secondly, this method returns True if the pair of points being compared
    fall within that distance threshold.
    """
    # Parse out the floating point coordinates.
    x = parse_coordinates(z[0][1])
    y = parse_coordinates(z[1][1])

    # All done!
    return la.norm(x - y) < EPSILON.value

def pairwise_pixels(pixels):
    """
    Computes the affinity for a pair of pixels.
    """
    i, j = pixels
    rbf = pairwise.rbf_kernel(IMAGE.value[i], IMAGE.value[j], gamma = SIGMA.value)[0, 0]
    return [(i, [j, rbf]), (j, [i, rbf])]

def pixel_row_vector(affinities):
    """
    Assembles the affinities into a correct row vector.
    """
    rowid, values = affinities
    return [rowid, {v[0]: v[1] for v in values}]

def pairwise_points(z):
    """
    Computes the RBF affinity for a pair of Cartesian points.
    """
    # Parse out floating point coordinates.
    x = parse_coordinates(z[0][1])
    y = parse_coordinates(z[1][1])

    # Find the RBF kernel between them.
    return [int(z[0][0]), [int(z[1][0]), pairwise.rbf_kernel(x, y, gamma = SIGMA.value)[0, 0]]]

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
        IMAGE = sc.broadcast(image.ravel())
        A = sc.parallelize(connectivity(image.shape[0], image.shape[1]), sc.defaultParallelism * 4)
    else:
        # Read the input file, index each data point, and parallelize to an RDD.
        rawdata = np.loadtxt(args['input'], dtype = np.str, delimiter = "\n")
        indexed = np.vstack([np.arange(rawdata.shape[0]), rawdata]).T
        EPSILON = sc.broadcast(args['epsilon'])
        D = sc.parallelize(indexed)
        A = D.cartesian(D)
        if EPSILON.value > 0.0:
            A = A.filter(distance_threshold)

    # If sigma was not specified, we'll compute it ourselves. We do this by first
    # finding the *median difference* between all points (connected pixels or Cartesian
    # data that passes the distance threshold), and use that to compute sigma.
    if sigma <= 0.0:
        d = np.median(np.array(
                A.map(
                    lambda x:
                        np.abs(IMAGE.value[x[0]] - IMAGE.value[x[1]])
                        if TYPE.value == 'image' else
                        la.norm(parse_coordinates(x[0][1]) - parse_coordinates(x[1][1]))
                     )
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
        affinities = A.map(pairwise_points).sortByKey().collect()

        print 'Of %sx%s possible pairs, we have %s.' % (rawdata.shape[0], rawdata.shape[0], len(affinities))

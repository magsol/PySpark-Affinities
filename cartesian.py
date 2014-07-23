import argparse
import numpy as np
import numpy.linalg as la
import sklearn.metrics.pairwise as pairwise

from pyspark import SparkConf, SparkContext

def pairwise_blocks(x):
    # Extract the indices and string form of the data.
    x1 = x[0]
    x2 = x[1]
    i1, data1 = int(x1[0]), x1[1]
    i2, data2 = int(x2[0]), x2[1]

    # Now parse out the floating point coordinates.
    x = np.array(map(float, data1.strip().split(",")))
    y = np.array(map(float, data2.strip().split(",")))

    # Find the RBF kernel between them, assuming their distance is within
    # the threshold.
    e = EPSILON.value
    s = SIGMA.value

    # Are these two points close enough together?
    threshold = la.norm(x - y) < e if e > 0.0 else True
    std = 1.0 / (2 * s * s)
    rbf = pairwise.rbf_kernel(x, y, gamma = std)[0, 0] if threshold else 0.0
    return [i1, [i2, rbf]]

    # We don't need to return a pair of tuples, because all pairings
    # are enumerated; there will be another case where the values of 
    # i1 and i2 are switched.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'PySpark Affinities',
        epilog = 'lol sp3c+r4l', add_help = 'How to use',
        prog = 'python cartesian.py <arguments>')

    # Required arguments.
    parser.add_argument("-i", "--input", required = True,
        help = "Path to a text file, where each line is a point in n-dimensional space (each dimension separated by a comma).")
    #parser.add_argument("-o", "--output", required = True,
    #    help = "Path to an output directory. If it doesn't exist, it is created.")

    # Optional arguments.
    parser.add_argument("-e", "--epsilon", type = float, default = 2.0,
        help = "If specified, pairwise affinities are only computed for points whose Euclidean distance is less than this. [DEFAULT: 0.0]")
    parser.add_argument("-s", "--sigma", type = float, default = 1.0,
        help = "If specified, the value of sigma to use in computing RBF affinities. [DEFAULT: 3.0]")

    args = vars(parser.parse_args())

    # Set up the Spark context. Because awesome.
    sc = SparkContext(conf = SparkConf())

    # Read the input file, index each data point, and parallelize to an RDD.
    rawdata = np.loadtxt(args['input'], dtype = np.str, delimiter = "\n")
    indexed = np.vstack([np.arange(rawdata.shape[0]), rawdata]).T
    #D = sc.parallelize(indexed, sc.defaultParallelism * 4)
    D = sc.parallelize(indexed)

    # Broadcast variables.
    SIGMA = sc.broadcast(args['sigma'])
    EPSILON = sc.broadcast(args['epsilon'])

    # Map over each pair of points.
    retval = D.cartesian(D).map(pairwise_blocks).filter(lambda x: x[1][1] > 0.0).collect()

    print 'Of %sx%s possible pairs, we have %s.' % (rawdata.shape[0], rawdata.shape[0], len(retval))

import numpy as np

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

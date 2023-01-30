import itertools as it
import numpy as np
from scipy.stats.mstats import mquantiles
from scipy import ndimage as nd
from scipy import sparse
from skimage import morphology as skmorph
from skimage import filters as filters, measure, util
from skimage.feature import greycomatrix, greycoprops
from skimage.util import img_as_int
from sklearn.neighbors import NearestNeighbors
import cytoolz as tz
from ._util import normalise_random_state


def normalize_vectors(v):
    """Interpret a matrix as a row of vectors, and divide each by its norm.

    0-vectors are left changed.

    Parameters
    ----------
    v : array of float, shape (M, N)
        M points of dimension N.

    Returns
    -------
    v1 : array of float, shape (M, N)
        The vectors divided by their norm.

    Examples
    --------
    >>> vs = np.array([[2., 0.], [0., 4.], [0., 0.]])
    >>> normalize_vectors(vs)
    array([[1., 0.],
           [0., 1.],
           [0., 0.]])
    """
    v_norm = np.sqrt((v ** 2).sum(axis=1))
    v_norm[v_norm == 0] = 1.  # ignore 0-vectors for division
    v1 = v / v_norm[..., np.newaxis]
    return v1


def triplet_angles(points, indices):
    """Compute the angles formed by point triplets.

    Parameters
    ----------
    points : array of float, shape (M, N)
        Set of M points in N-dimensional space.
    indices : array of int, shape (Q, 3)
        Set of Q index triplets, in order (root, leaf1, leaf2). Thus,
        the angle is computed between the vectors
            (points[leaf1] - points[root])
        and
            (points[leaf2] - points[root]).

    Returns
    -------
    angles : array of float, shape (Q,)
        The desired angles.
    """
    angles = np.zeros(len(indices), np.double)
    roots = points[indices[:, 0]]
    leaf1 = points[indices[:, 1]]
    leaf2 = points[indices[:, 2]]
    u = normalize_vectors(leaf1 - roots)
    v = normalize_vectors(leaf2 - roots)
    cosines = (u * v).sum(axis=1)
    cosines[cosines > 1] = 1
    cosines[cosines < -1] = -1
    angles = np.arccos(cosines)
    return angles


def nearest_neighbors(lab_im, n=3, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]):
    """Find the distances to and angle between the n nearest neighbors.

    Parameters
    ----------
    lab_im : 2D array of int
        An image of labeled objects.
    n : int, optional
        How many nearest neighbors to check. (Angle is always between
        the two nearest only.)
    quantiles : list of float in [0, 1], optional
        Which quantiles of the features to compute.

    Returns
    -------
    nei : 1D array of float, shape (5 * (n + 1),)
        The quantiles of sines, cosines, angles, and `n` nearest neighbor
        distances.
    names : list of string
        The name of each feature.
    """
    if lab_im.dtype == bool:
        lab_im = nd.label(lab_im)[0]
    centroids = np.array(
        [p.centroid for p in measure.regionprops(lab_im, coordinates='rc')])
    nbrs = (NearestNeighbors(
        n_neighbors=(n + 1), algorithm='kd_tree').fit(centroids))
    distances, indices = nbrs.kneighbors(centroids)
    angles = triplet_angles(centroids, indices[:, :3])
    # ignore order/orientation of vectors, only measure acute angles
    angles[angles > np.pi] = 2 * np.pi - angles[angles > np.pi]
    distances[:, 0] = angles
    sines, cosines = np.sin(angles), np.cos(angles)
    features = np.hstack((sines[:, np.newaxis], cosines[:, np.newaxis],
                          distances))
    nei = mquantiles(features, quantiles, axis=0).ravel()
    colnames = (['sin-theta', 'cos-theta', 'theta'] +
                ['d-neighbor-%i-' % i for i in range(1, n + 1)])
    names = ['%s-percentile-%i' % (colname, int(q * 100))
             for colname, q in it.product(colnames, quantiles)]
    return nei, names


# threshold and labeling number of objects, statistics about object size and
# shape
def intensity_object_features(im, threshold=None, adaptive_t_radius=51,
                              sample_size=None, random_seed=None):
    """Segment objects based on intensity threshold and compute properties.

    Parameters
    ----------
    im : 2D np.ndarray of float or uint8.
        The input image.
    threshold : float, optional
        A threshold for the image to determine objects: connected pixels
        above this threshold will be considered objects. If ``None``
        (default), the threshold will be automatically determined with
        both Otsu's method and a locally adaptive threshold.
    adaptive_t_radius : int, optional
        The radius to calculate background with adaptive threshold.
    sample_size : int, optional
        Sample this many objects randomly, rather than measuring all
        objects.
    random_seed: int, or numpy RandomState instance, optional
        An optional random number generator or seed from which to draw
        samples.

    Returns
    -------
    f : 1D np.ndarray of float
        The feature vector.
    names : list of string
        The list of feature names.
    """
    if threshold is None:
        tim1 = im > filters.threshold_otsu(im)
        f1, names1 = object_features(tim1, im, sample_size=sample_size,
                                     random_seed=random_seed)
        names1 = ['otsu-threshold-' + name for name in names1]
        tim2 = im > filters.threshold_local(im, adaptive_t_radius)
        f2, names2 = object_features(tim2, im, sample_size=sample_size,
                                     random_seed=random_seed)
        names2 = ['adaptive-threshold-' + name for name in names2]
        f = np.concatenate([f1, f2])
        names = names1 + names2
    else:
        tim = im > threshold
        f, names = object_features(tim, im, sample_size=sample_size,
                                   random_seed=random_seed)
    return f, names


def object_features(bin_im, im, erode=2, sample_size=None, random_seed=None):
    """Compute features about objects in a binary image.

    Parameters
    ----------
    bin_im : 2D np.ndarray of bool
        The image of objects.
    im : 2D np.ndarray of float or uint8
        The actual image.
    erode : int, optional
        Radius of erosion of objects.
    sample_size : int, optional
        Sample this many objects randomly, rather than measuring all
        objects.
    random_seed: int, or numpy RandomState instance, optional
        An optional random number generator or seed from which to draw
        samples.

    Returns
    -------
    fs : 1D np.ndarray of float
        The feature vector.
    names : list of string
        The names of each feature.
    """
    random = normalise_random_state(random_seed)
    selem = skmorph.disk(erode)
    if erode > 0:
        bin_im = nd.binary_opening(bin_im, selem)
    lab_im, n_objs = nd.label(bin_im)
    if sample_size is None:
        sample_size = n_objs
        sample_indices = np.arange(n_objs)
    else:
        sample_indices = random.randint(0, n_objs, size=sample_size)
    prop_names = ['area', 'eccentricity', 'euler_number', 'extent',
                  'min_intensity', 'mean_intensity', 'max_intensity',
                  'solidity']
    objects = measure.regionprops(lab_im, intensity_image=im, coordinates='rc')
    properties = np.empty((sample_size, len(prop_names)), dtype=np.float)
    for i, j in enumerate(sample_indices):
        properties[i] = [getattr(objects[j], prop) for prop in prop_names]
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    feature_quantiles = mquantiles(properties, quantiles, axis=0).T
    fs = np.concatenate([np.array([n_objs], np.float),
                         feature_quantiles.ravel()])
    names = (['num-objs'] +
             ['%s-percentile%i' % (prop, int(q * 100))
              for prop, q in it.product(prop_names, quantiles)])
    return fs, names


def haralick_features(im, prop_names=None, distances=[2, 4, 8], angles=np.arange(8) * np.pi/4,
                      levels=256, symmetric=False, normed=False):
    """Compute Haralick texture features of a grayscale image.

    Parameters
    ----------
    im : 2D np.ndarray of float or uint8.
        The input image.
    prop_names : list of strings, optional
        Texture properties of a gray level co-occurence matrix.
        By default prop_names=None, which means all properties are computed.
        Available texture properties include: 'contrast', 'dissimilarity',
        'homogeneity', 'ASM', 'energy', and 'correlation'.
    distances : array_like, optional
        List of pixel pair distance offsets, used for grey covariance matrix.
    angles : array_like, optional
        List of pixel pair angles in radians, used for grey covariance matrix.
    levels : int, optional
        The input image should contain integers in [0, levels-1],
        where levels indicate the number of grey-levels counted
        (typically 256 for an 8-bit image).
        This argument is required for 16-bit images or higher and is typically
        the maximum of the image. As the output matrix is at least
        levels x levels, it might be preferable to use binning of the
        input image rather than large values for levels.
    symmetric : bool, optional
        If True, the output matrix P[:, :, d, theta] is symmetric.
        This is accomplished by ignoring the order of value pairs,
        so both (i, j) and (j, i) are accumulated when (i, j)
        is encountered for a given offset. The default is False.
    normed : bool, optional
        If True, normalize each matrix P[:, :, d, theta] by dividing by
        the total number of accumulated co-occurrences for the given offset.
        The elements of the resulting matrix sum to 1. The default is False.

    Returns
    -------
    fs : 1D np.ndarray of float
        The feature vector.
    names : list of string
        The list of feature names.

    References
    ----------
    .. [1] The GLCM Tutorial Home Page,
           http://www.fp.ucalgary.ca/mhallbey/tutorial.htm
    """
    if np.issubdtype(im.dtype, np.floating):
        im = img_as_int(im)

    available_prop_names = ['contrast',
                            'dissimilarity',
                            'homogeneity',
                            'ASM',
                            'energy',
                            'correlation']
    if prop_names is None:
        prop_names = available_prop_names
    else:  # do not allow invalid input in prop_names
        prop_names = [prop for prop in prop_names
                      if prop.lower() in map(str.lower, available_prop_names)]
    glcm = greycomatrix(im, distances=distances, angles=angles,
                        levels=levels, symmetric=symmetric, normed=normed)
    fs = []
    names = []
    for prop in prop_names:
        texture_properties = greycoprops(glcm, prop)
        for dist, theta in it.product(distances, angles):
            name = 'haralick-%s-distance%d-angle%d' % (prop, dist, theta)
            names.append(name)
            fs.append(texture_properties[distances.index(dist),
                                         angles.index(theta)])

    return np.array(fs), names


def fraction_positive(bin_im, positive_im, erode=2, overlap_thresh=0.9,
                      bin_name='nuclei', positive_name='tf'):
    """Compute fraction of objects in bin_im overlapping positive_im.

    The purpose of this function is to compute the fraction of nuclei
    that express a particular transcription factor. By providing the
    thresholded DAPI channel as `bin_im` and the thresholded TF channel
    as `positive_im`, this fraction can be computed.

    Parameters
    ----------
    bin_im : 2D array of bool
        The image of objects being tested.
    positive_im : 2D array of bool
        The image of positive objects.
    erode : int, optional
        Radius of structuring element used to smooth input images.
    overlap_thresh : float, optional
        The minimum amount of overlap between an object in `bin_im` and
        the `positive_im` to consider that object "positive".
    bin_name : string, optional
        The name of the objects being tested.
    positive_name : string, optional
        The name of the property being measured.

    Returns
    -------
    f : 1D array of float, shape (1,)
        The feature vector.
    name : list of string, length 1
        The name of the feature.

    Examples
    --------
    >>> bin_im = np.array([[1, 1, 0],
    ...                    [0, 0, 0],
    ...                    [1, 1, 1]], dtype=bool)
    >>> pos_im = np.array([[1, 0, 0],
    ...                    [0, 1, 1],
    ...                    [0, 1, 1]], dtype=bool)
    >>> f = fraction_positive(bin_im, pos_im, erode=0, overlap_thresh=0.6)
    >>> f[0]
    array([0.5])
    >>> f[1][0]
    'frac-nuclei-pos-tf-erode-0-thresh-0.60'
    """
    selem = skmorph.disk(erode)
    if erode > 0:
        bin_im = nd.binary_opening(bin_im, selem)
        positive_im = nd.binary_opening(positive_im, selem)
    lab_im = nd.label(bin_im)[0].ravel()
    pos_im = positive_im.ravel().astype(int)
    counts = sparse.coo_matrix((np.ones(lab_im.size),
                                (lab_im, pos_im))).toarray()
    means = counts[:, 1] / np.sum(counts, axis=1)
    f = np.array([np.mean(means[1:] > overlap_thresh)])
    name = ['frac-%s-pos-%s-erode-%i-thresh-%.2f' %
            (bin_name, positive_name, erode, overlap_thresh)]
    return f, name


def nuclei_per_cell_histogram(nuc_im, cell_im, max_value=10):
    """Compute the histogram of nucleus count per cell object.

    Counts above or below max_value and min_value are clipped.

    Parameters
    ----------
    nuc_im : array of bool or int
        An image of nucleus objects, binary or labelled.
    cell_im : array of bool or int
        An image of cell objects, binary or labelled.
    max_value : int, optional
        The highest nucleus count we expect. Anything above this will
        be clipped to ``max_value + 1``.

    Returns
    -------
    fs : array of float, shape ``(max_value - min_value + 2,)``.
        The proportion of cells with each nucleus counts.
    names : list of string, same length as fs
        The name of each feature.
    """
    names = [('cells-with-%i-nuclei' % n) for n in range(max_value + 1)]
    names.append('cells-with->%i-nuclei' % max_value)
    nuc_lab = nd.label(nuc_im)[0]
    cell_lab = nd.label(cell_im)[0]
    match = np.vstack((nuc_lab.ravel(), cell_lab.ravel())).T
    match = match[(match.sum(axis=1) != 0), :]
    match = util.unique_rows(match).astype(np.int64)
    # number of nuclei in each cell
    cells = np.bincount(match[:, 1])
    # number of cells with x nuclei
    nhist = np.bincount(cells, minlength=max_value + 2)
    total = np.sum(nhist)
    fs = np.zeros((max_value + 2), np.float)
    fs[:(max_value + 1)] = nhist[:(max_value + 1)]
    fs[max_value + 1] = np.sum(nhist[(max_value + 1):])
    fs /= total
    return fs, names


@tz.curry
def default_feature_map(image, threshold=None,
                        channels=[0, 1, 2], channel_names=None,
                        sample_size=None, random_seed=None):
    """Compute a feature vector from a multi-channel image.

    Parameters
    ----------
    image : array, shape (M, N, 3)
        The input image.
    threshold : tuple of float, shape (3,), optional
        The intensity threshold for object detection on each channel.
    channels : list of int, optional
        Which channels to use for feature computation
    channel_names : list of string, optional
        The channel names corresponding to ``channels``.
    sample_size : int, optional
        For features based on quantiles, sample this many objects
        rather than computing full distribution. This can considerably
        speed up computation with little cost to feature accuracy.
    random_seed: int, or numpy RandomState instance, optional
        An optional random number generator or seed from which to draw
        samples.

    Returns
    -------
    fs : 1D array of float
        The features of the image.
    names : list of string
        The feature names.
    """
    all_fs, all_names = [], []
    images = [np.array(image[..., i]) for i in channels]
    if channel_names is None:
        channel_names = ['chan{}'.format(i) for i in channels]
    for im, prefix in zip(images, channel_names):
        fs, names = intensity_object_features(im, threshold=threshold,
                                              sample_size=sample_size,
                                              random_seed=random_seed)
        names = [prefix + '-' + name for name in names]
        all_fs.append(fs)
        all_names.extend(names)
    return np.concatenate(all_fs), all_names

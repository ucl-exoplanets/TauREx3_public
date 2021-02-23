import numpy as np


def normalize(v, axis=0):
    norm = np.linalg.norm(v, axis=axis)
    solution = v/norm
    solution[:, norm == 0] = v[:, norm == 0]
    return solution


def sphere_to_cartesian(R, vec, degrees=True):
    result = np.zeros_like(vec)
    _vec = vec[...]
    if degrees:
        _vec[1:] = np.radians(_vec[1:])
    result[0] = (R+_vec[0])*np.sin(_vec[1])*np.cos(_vec[2])
    result[1] = (R+_vec[0])*np.sin(_vec[1])*np.sin(_vec[2])
    result[2] = (R+_vec[0])*np.cos(_vec[1])
    return result

def compute_line_3d(v, t, axis=0):
    """
    Generates a line given two cartesian points
    """
    v = np.array(v)
    t = np.array(t)
    o = v[...]
    u = normalize(t-v, axis=axis)
    return o, u


def parallel_vector(R, alt, max_alt=1e5):
    """
    Generates a viewing and tangent vectors
    parallel to the surface of a sphere
    """

    if not hasattr(alt, '__len__'):
        alt = np.array([alt])
    viewer = np.zeros(shape=(3, len(alt)))
    tangent = np.zeros_like(viewer)
    viewer[0] = -(R+max_alt*2)
    viewer[1] = R+alt
    tangent[1] = R+alt

    return viewer, tangent


def perpendicular_vector(R, max_alt):
    """
    Generates a viewing and tangent vectors
    perpendicular to a sphere of radius R
    """
    return np.array([0, R+max_alt*2, 0]).reshape(3, 1), \
        np.array([0.0, 0, 0]).reshape(3, 1)

def multi_dot(a, b):
    return np.sum(a*b, axis=0)

def compute_intersection_3d(R, h, u, o, c=np.zeros(3), allow_single=False):
    """
    Computes line-sphere intersection for a range of radius offsets
    Will detect if path crosses R and cutoff the point

    Parameters
    ----------
    R: float
        Radius

    h: float or array_like of shape (m)
        height above radius
    
    u: array with shape (3,n)
        normalized direction vector

    o: array with shape (3,n)
        origin vector

    c: vector, optional
        center position of sphere default is (0,0,0)

    Returns
    --------
    solution: array of shape (2,3,m,n)
        Array containing points that intersect the sphere
        First index is always the point closest to origin vector


    """
    if not hasattr(h, '__len__'):
        h = np.array([h])
    else:
        h = np.array(h)
    
    if len(u.shape) == 1:
        u = u.reshape(-1, 1)
        o = o.reshape(-1, 1)
    
    single_dot = multi_dot(u, o)
    dot_res = (single_dot)**2 - (o**2).sum(axis=0)
    delta = dot_res + (R+h[..., None])**2
    solution = np.zeros(shape=(2, 3, h.shape[0], u.shape[1]))
    sol1 = np.zeros(shape=(3, h.shape[0], u.shape[1]))
    sol2 = np.zeros(shape=(3, h.shape[0], u.shape[1]))
    solution[...] = np.nan
    filt = delta > 0
    if allow_single:
        filt = delta >= 0
    if filt.sum() == 0:
        return None
    
    with np.errstate(divide='ignore', invalid='ignore'):

        d1 = -(single_dot) + np.sqrt(delta)
        d1[d1 < 0] = 0.0
        sol1 = o[:, None, :] + d1*u[:, None]

        d2 = -(single_dot) - np.sqrt(delta)
        d2[d2 < 0] = 0.0
        sol2 = o[:, None, :] + d2*u[:, None]
        v1 = np.sum((o[:, None, :]-sol1)**2, axis=0)
        v2 = np.sum((o[:, None, :]-sol2)**2, axis=0)
    #
    #print(v1.shape)
    max_filter = v2 > v1
    #print(max_filter.shape)
    a_filter = max_filter
    b_filter = ~max_filter
    #print(solution.shape)
    if a_filter.sum() > 0:
        solution[0, :, a_filter] = sol1[:, a_filter].T
        solution[1, :, a_filter] = sol2[:, a_filter].T
    if b_filter.sum() > 0:
        solution[0, :, b_filter] = sol2[:, b_filter].T

        solution[1, :, b_filter] = sol1[:, b_filter].T
    
    # Detect planet crossings
    delta = dot_res + (R)**2

    filt = delta > 0
    tang = np.where(filt)[0]
    if filt.sum() > 0:
        d1 = -(single_dot[filt]) + np.sqrt(delta[filt])
        sol1 = o[:, filt] + d1*u[:, filt]
        d2 = -(single_dot[filt]) - np.sqrt(delta[filt])
        sol2 = o[:, filt] + d2*u[:, filt]
        v1 = ((o[:, filt]-sol1)**2).sum()
        v2 = ((o[:, filt]-sol2)**2).sum()
        if v2 > v1:
            solution[1, ..., tang] = sol1.T[..., None]
        else:
            solution[1, ..., tang] = sol2.T[..., None]

    return solution


def compute_path_length_3d(R, altitudes, viewer, tangent,
                           coordinates='cartesian'):
    """
    Given a viewing and tangent vector, computes the path length for a sphere
    """

    _viewer = viewer[...]
    _tangent = tangent[...]

    if not hasattr(altitudes, '__len__'):
        altitudes = np.array([altitudes])
    else:
        altitudes = np.array(altitudes)

    if not isinstance(coordinates, (list, tuple,)):

        coordinates = [coordinates, coordinates]

    if coordinates[0] in ('spherical'):
        _viewer = sphere_to_cartesian(R, _viewer)
    if coordinates[1] in ('spherical'):
        _tangent = sphere_to_cartesian(R, _tangent)

    if len(_viewer.shape) == 1:
        _viewer = _viewer.reshape(-1, 1)
        _tangent = _viewer.reshape(-1, 1)

    o, u = compute_line_3d(_viewer,
                           _tangent)
    intersections = compute_intersection_3d(R, altitudes, u, o)

    if intersections is not None:

        distances = (np.linalg.norm(intersections[1] - intersections[0],
                                    axis=0))
        filt = np.isfinite(distances)
        all_distances = []

        for i in range(_viewer.shape[1]):
            layer_filt = filt[:, i]
            good_indices = np.where(layer_filt)[0]
            dists = distances[layer_filt, i]
            final_distances = np.zeros_like(dists)
            final_distances[0] = dists[0]
            final_distances[1:] = dists[1:]-dists[:-1]

            all_distances.append((good_indices, final_distances))
        return all_distances
    else:
        return None
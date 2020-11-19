import numpy as np


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm


def compute_line_3d(v, t):
    """
    Generates a line given two cartesian points
    """
    v = np.array(v)
    t = np.array(t)
    o = v[...]
    u = normalize(t-v)
    return o, u


def parallel_vector(R, alt, max_alt):
    return np.array([-R+max_alt*2, R+alt, 0]), np.array([0.0, R+alt, 0])


def perpendicular_vector(R,max_alt):
    return np.array([0, R+max_alt*2, 0]),np.array([0.0, 0, 0])


def compute_intersection_3d(R, h, u, o, c=np.zeros(3)):
    """
    Computes line-sphere intersection for a range of radius offsets
    Will detect if path crosses R and cutoff the point

    Parameters
    ----------
    R: float
        Radius
    
    h: float or array_like
        height above radius
    
    u: vector
        normalized direction vector
    
    o: vector
        origin vector
    
    c: vector, optional
        center position of sphere default is (0,0,0)
    
    Returns
    --------
    solution: array of shape (2,3,len(h))
        Array containing points that intersect the sphere
        First index is always the point closest to origin vector


    """

    if not hasattr(h, '__len__'):
        h = np.array([h])
    else:
        h = np.array(h)

    dot_res = (np.dot(u, o))**2 - (o**2).sum()
    delta = dot_res + (R+h)**2
    solution = np.zeros(shape=(2, 3, h.shape[0]))
    sol1 = np.zeros(shape=(3, h.shape[0]))
    sol2 = np.zeros(shape=(3, h.shape[0]))
    solution[...] = np.nan
    filt = delta > 0
    if np.all(~filt):
        return None
    d1 = -(np.dot(u, o-c)) + np.sqrt(delta[filt])
    sol1[:, filt] = o[:, None] + d1*u[:, None]
    d2 = -(np.dot(u, o-c)) - np.sqrt(delta[filt])
    sol2[:, filt] = o[:, None] + d2*u[:, None]
    v1 = np.sum((o[:, None]-sol1)**2, axis=0)
    v2 = np.sum((o[:, None]-sol2)**2, axis=0)
    max_filter = v2 > v1

    a_filter = filt & max_filter
    b_filter = filt & ~max_filter
    if a_filter.sum() > 0:
        solution[0, :, a_filter] = sol1[:, a_filter].T
        solution[1, :, a_filter] = sol2[:, a_filter].T
    if b_filter.sum() > 0:
        solution[0, :, b_filter] = sol2[:, b_filter].T

        solution[1, :, b_filter] = sol1[:, b_filter].T

    # Detect planet crossings
    delta = dot_res + (R)**2
    if delta > 0:
        d1 = -(np.dot(u, o-c)) + np.sqrt(delta)
        sol1 = o + d1*u
        d2 = -(np.dot(u, o-c)) - np.sqrt(delta)
        sol2 = o + d2*u
        v1 = ((o-sol1)**2).sum()
        v2 = ((o-sol2)**2).sum()
        if v2 > v1:
            solution[1, ...] = sol1[:, None]
        else:
            solution[1, ...] = sol2[:, None]
            
        
    return solution
        
def compute_path_length_3d(R, altitudes, viewer, tangent,coordinates='cartesian'):
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
    
    o, u = compute_line_3d(_viewer,
                           _tangent)
    intersections = compute_intersection_3d(R, altitudes, u, o)
    good_solutions = np.isfinite(intersections[0,0,:])
#     print(good_solutions)
#     print(solution)
    good_indices = np.where(good_solutions)[0]
    intersections = intersections[..., good_solutions]
    distances = np.linalg.norm(intersections[1,...]-intersections[0,...],axis=0)
    
    final_distances = np.zeros_like(distances)
    final_distances[0] = distances[0]
    final_distances[1:] = distances[1:]-distances[:-1]
    
    
    return good_indices, final_distances
    
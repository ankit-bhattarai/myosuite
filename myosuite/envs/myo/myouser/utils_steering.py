import functools
import numpy as np
import scipy

def cross2d(x, y):
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

def heading_angle(parametrization_fct, theta, delta=1e-5, angle_change=False, rel_vec=np.array([1, 0])
):
    if (isinstance(theta, np.ndarray) and theta.ndim > 0) or isinstance(theta, list):
        return np.vectorize(functools.partial(heading_angle, parametrization_fct=parametrization_fct, delta=delta))(theta=np.array(theta))
    
    if angle_change:
        # compute change in heading angle
        _fwd_delta, _bwd_delta = parametrization_fct(theta + delta) - parametrization_fct(theta), parametrization_fct(theta) - parametrization_fct(max(theta - delta, 0))
        angle = np.arccos(np.dot(_fwd_delta, -_bwd_delta)/(np.linalg.norm(_fwd_delta)*np.linalg.norm(_bwd_delta)))
    else:
        # compute heading angle relative to horizontal axis (x-axis in these internal computations)
        _fwd_delta, _bwd_delta = parametrization_fct(theta + delta) - parametrization_fct(theta), parametrization_fct(theta) - parametrization_fct(max(theta - delta, 0))
        if theta == 0:
            _bwd_delta = _fwd_delta
        # fwd_angle = np.arccos(np.dot(_fwd_delta, rel_vec)/(np.linalg.norm(_fwd_delta)*np.linalg.norm(rel_vec))) % np.pi
        # bwd_angle = np.arccos(np.dot(-_bwd_delta, rel_vec)/(np.linalg.norm(-_bwd_delta)*np.linalg.norm(rel_vec))) % np.pi
        fwd_angle = (np.arctan2(-(cross2d(_fwd_delta, rel_vec)), np.dot(_fwd_delta, rel_vec)) + np.pi) % (2*np.pi) - np.pi
        bwd_angle = (np.arctan2(-(cross2d(_bwd_delta, rel_vec)), np.dot(_bwd_delta, rel_vec)) + np.pi) % (2*np.pi) - np.pi
    return fwd_angle, bwd_angle

def angle_to_rot_matrix(angle):
    if (isinstance(angle, np.ndarray) and angle.ndim > 0) or isinstance(angle, list):
        return np.vectorize(functools.partial(angle_to_rot_matrix), otypes=[np.ndarray])(angle=np.array(angle))
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

def heading_rotation_matrix(parametrization_fct, theta, delta=1e-5):
    fwd_angle, bwd_angle = heading_angle(parametrization_fct, theta, delta=delta, angle_change=False)
    fwd_rot_matrix = angle_to_rot_matrix(fwd_angle)
    bwd_rot_matrix = angle_to_rot_matrix(bwd_angle)
    return fwd_rot_matrix, bwd_rot_matrix

def tunnel_from_nodes(nodes, tunnel_size=None, ord=np.inf, width_height_constraints=None):
    assert tunnel_size is not None or width_height_constraints is not None
    
    if (isinstance(tunnel_size, np.ndarray) and len(tunnel_size) > 0) or isinstance(tunnel_size, list):
        assert len(tunnel_size) == len(nodes), "If multiple tunnel sizes are provided, the length of 'tunnel_size' needs to match the length of 'nodes'."
        tunnel_size = np.array(tunnel_size).reshape(-1, 1)
    else:
        tunnel_size = np.repeat(tunnel_size, len(nodes)).reshape(-1, 1)

    theta = np.linspace(0, 1, len(nodes))
    _interp_fct = scipy.interpolate.make_interp_spline(theta, nodes, k=1)

    fwd_angle, bwd_angle = heading_angle(_interp_fct, theta=theta, angle_change=False)
    fwd_rot_matrix = angle_to_rot_matrix(fwd_angle)
    bwd_rot_matrix = angle_to_rot_matrix(bwd_angle)

    _bwd_offset_left = np.vstack(list(map(lambda x: x.dot(angle_to_rot_matrix(np.pi/2).dot(np.array([1, 0]))), bwd_rot_matrix)))
    _fwd_offset_left = np.vstack(list(map(lambda x: x.dot(angle_to_rot_matrix(np.pi/2).dot(np.array([1, 0]))), fwd_rot_matrix)))
    _outer_offset_left = (_bwd_offset_left + _fwd_offset_left) * (_bwd_offset_left != _fwd_offset_left) + _fwd_offset_left * (_bwd_offset_left == _fwd_offset_left)
    if width_height_constraints is not None:
        for id, (con_type, con) in enumerate(width_height_constraints):
            if con_type == "width":
                _outer_offset_left[id, 0] = 0.5 * con * (np.sign(_outer_offset_left[id, 0]) if np.sign(_outer_offset_left[id, 0]) != 0 else 1)
                if id > 0 and (width_height_constraints[id-1][0] == "height"):
                    # apply last height constraint again
                    last_con = width_height_constraints[id-1][1]
                    _outer_offset_left[id, 1] = 0.5 * last_con * (np.sign(_outer_offset_left[id, 1]) if np.sign(_outer_offset_left[id, 1]) != 0 else 1)
            elif con_type == "height":
                _outer_offset_left[id, 1] = 0.5 * con * (np.sign(_outer_offset_left[id, 1]) if np.sign(_outer_offset_left[id, 1]) != 0 else 1)
                if id > 0 and (width_height_constraints[id-1][0] == "width"):
                    # apply last width constraint again
                    last_con = width_height_constraints[id-1][1]
                    _outer_offset_left[id, 0] = 0.5 * last_con * (np.sign(_outer_offset_left[id, 0]) if np.sign(_outer_offset_left[id, 0]) != 0 else 1)
    else:
        _outer_offset_left = 0.5*tunnel_size*_outer_offset_left/np.linalg.norm(_outer_offset_left, axis=1, ord=ord).reshape(-1, 1)
    nodes_left = nodes + _outer_offset_left

    _bwd_offset_right = np.vstack(list(map(lambda x: x.dot(angle_to_rot_matrix(-np.pi/2).dot(np.array([1, 0]))), bwd_rot_matrix)))
    _fwd_offset_right = np.vstack(list(map(lambda x: x.dot(angle_to_rot_matrix(-np.pi/2).dot(np.array([1, 0]))), fwd_rot_matrix)))
    _outer_offset_right = (_bwd_offset_right + _fwd_offset_right) * (_bwd_offset_right != _fwd_offset_right) + _fwd_offset_right * (_bwd_offset_right == _fwd_offset_right)
    if width_height_constraints is not None:
        for id, (con_type, con) in enumerate(width_height_constraints):
            if con_type == "width":
                _outer_offset_right[id, 0] = 0.5 * con * (np.sign(_outer_offset_right[id, 0]) if np.sign(_outer_offset_right[id, 0]) != 0 else 1)
                if id > 0 and (width_height_constraints[id-1][0] == "height"):
                    # apply last height constraint again
                    last_con = width_height_constraints[id-1][1]
                    _outer_offset_right[id, 1] = 0.5 * last_con * (np.sign(_outer_offset_right[id, 1]) if np.sign(_outer_offset_right[id, 1]) != 0 else 1)
            elif con_type == "height":
                _outer_offset_right[id, 1] = 0.5 * con * (np.sign(_outer_offset_right[id, 1]) if np.sign(_outer_offset_right[id, 1]) != 0 else 1)
                if id > 0 and (width_height_constraints[id-1][0] == "width"):
                    # apply last width constraint again
                    last_con = width_height_constraints[id-1][1]
                    _outer_offset_right[id, 0] = 0.5 * last_con * (np.sign(_outer_offset_right[id, 0]) if np.sign(_outer_offset_right[id, 0]) != 0 else 1)
    else:
        _outer_offset_right = 0.5*tunnel_size*_outer_offset_right/np.linalg.norm(_outer_offset_right, axis=1, ord=ord).reshape(-1, 1)
    nodes_right = nodes + _outer_offset_right

    return nodes_left, nodes_right

def distance_to_tunnel(test_point, _interp_fct_left, _interp_fct_right):
    """
    Estimates(!) whether a given point is inside tunnel or not, and returns the signed distance (negative distance: outside tunnel).
    """
    # find nearest parametrization value of both boundary curves
    # theta_closest = scipy.optimize.minimize(lambda theta: np.linalg.norm(_interp_fct(theta) - test_point), 0.7, bounds=[(0, 1)], method="Nelder-Mead", options={"maxiter": 100000}).x.item()
    theta_closest_left = scipy.optimize.shgo(lambda theta: np.linalg.norm(_interp_fct_left(theta) - test_point), bounds=[(0, 1)]).x.item()
    theta_closest_right = scipy.optimize.shgo(lambda theta: np.linalg.norm(_interp_fct_right(theta) - test_point), bounds=[(0, 1)]).x.item()
    left_bound_closest, right_bound_closest = _interp_fct_left(theta_closest_left), _interp_fct_right(theta_closest_right)
    left_vector = left_bound_closest - test_point
    right_vector = right_bound_closest - test_point
    inside_tunnel = np.dot(left_bound_closest - test_point, right_bound_closest - test_point) < 0
    tunnel_distance = np.minimum(np.linalg.norm(left_vector), np.linalg.norm(right_vector)) * (-1)**(~inside_tunnel)
    return tunnel_distance, theta_closest_left, theta_closest_right
import functools
import jaxopt
import numpy as np
import scipy
import jax
import jax.numpy as jp
import interpax

def cross2d(x, y):
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

def heading_angle(parametrization_fct, theta, delta=1e-5, angle_change=False, rel_vec=jp.array([1, 0])
):
    if (isinstance(theta, jax.Array) and theta.ndim > 0) or isinstance(theta, list):
        return jax.vmap(functools.partial(heading_angle, parametrization_fct=parametrization_fct, delta=delta))(theta=jp.array(theta))
    
    if angle_change:
        # compute change in heading angle
        _fwd_delta, _bwd_delta = parametrization_fct(theta + delta) - parametrization_fct(theta), parametrization_fct(theta) - parametrization_fct(jp.maximum(theta - delta, 0))
        angle = jp.arccos(jp.dot(_fwd_delta, -_bwd_delta)/(jp.linalg.norm(_fwd_delta)*jp.linalg.norm(_bwd_delta)))
    else:
        # compute heading angle relative to horizontal axis (x-axis in these internal computations)
        _fwd_delta, _bwd_delta = parametrization_fct(theta + delta) - parametrization_fct(theta), parametrization_fct(theta) - parametrization_fct(jp.maximum(theta - delta, 0))
        _bwd_delta = jp.where(theta == 0, _fwd_delta, _bwd_delta)
        # fwd_angle = jp.arccos(jp.dot(_fwd_delta, rel_vec)/(jp.linalg.norm(_fwd_delta)*jp.linalg.norm(rel_vec))) % jp.pi
        # bwd_angle = jp.arccos(jp.dot(-_bwd_delta, rel_vec)/(jp.linalg.norm(-_bwd_delta)*jp.linalg.norm(rel_vec))) % jp.pi
        fwd_angle = (jp.arctan2(-(cross2d(_fwd_delta, rel_vec)), jp.dot(_fwd_delta, rel_vec)) + jp.pi) % (2*jp.pi) - jp.pi
        bwd_angle = (jp.arctan2(-(cross2d(_bwd_delta, rel_vec)), jp.dot(_bwd_delta, rel_vec)) + jp.pi) % (2*jp.pi) - jp.pi
    return fwd_angle, bwd_angle

def angle_to_rot_matrix(angle):
    if (isinstance(angle, jax.Array) and angle.ndim > 0) or isinstance(angle, list):
        return jax.vmap(functools.partial(angle_to_rot_matrix))(angle=jp.array(angle))
    return jp.array([[jp.cos(angle), -jp.sin(angle)], [jp.sin(angle), jp.cos(angle)]])

def heading_rotation_matrix(parametrization_fct, theta, delta=1e-5):
    fwd_angle, bwd_angle = heading_angle(parametrization_fct, theta, delta=delta, angle_change=False)
    fwd_rot_matrix = angle_to_rot_matrix(fwd_angle)
    bwd_rot_matrix = angle_to_rot_matrix(bwd_angle)
    return fwd_rot_matrix, bwd_rot_matrix

def tunnel_from_nodes(nodes, tunnel_size=None, ord=jp.inf, width_height_constraints=None):
    assert tunnel_size is not None or width_height_constraints is not None
    
    if (isinstance(tunnel_size, jax.Array) and len(tunnel_size) > 0) or isinstance(tunnel_size, list):
        assert len(tunnel_size) == len(nodes), "If multiple tunnel sizes are provided, the length of 'tunnel_size' needs to match the length of 'nodes'."
        tunnel_size = jp.array(tunnel_size).reshape(-1, 1)
    elif tunnel_size is not None:
        tunnel_size = jp.repeat(tunnel_size, len(nodes)).reshape(-1, 1)

    theta = jp.linspace(0, 1, len(nodes))
    _interp_fct = interpax.PchipInterpolator(theta, jp.array(nodes), check=False) #, k=1)

    fwd_angle, bwd_angle = heading_angle(_interp_fct, theta=theta, angle_change=False)
    fwd_rot_matrix = angle_to_rot_matrix(fwd_angle)
    bwd_rot_matrix = angle_to_rot_matrix(bwd_angle)

    _bwd_offset_left = jp.vstack(list(map(lambda x: x.dot(angle_to_rot_matrix(jp.pi/2).dot(jp.array([1, 0]))), bwd_rot_matrix)))
    _fwd_offset_left = jp.vstack(list(map(lambda x: x.dot(angle_to_rot_matrix(jp.pi/2).dot(jp.array([1, 0]))), fwd_rot_matrix)))
    _outer_offset_left = (_bwd_offset_left + _fwd_offset_left) * (_bwd_offset_left != _fwd_offset_left) + _fwd_offset_left * (_bwd_offset_left == _fwd_offset_left)
    if width_height_constraints is not None:
        for id, (con_type, con) in enumerate(width_height_constraints):
            if con_type == "width":
                _outer_offset_left = _outer_offset_left.at[id, 0].set(0.5 * con * (jp.sign(_outer_offset_left[id, 0]) * (jp.sign(_outer_offset_left[id, 0]) != 0) + 1 * jp.sign(_outer_offset_left[id, 0]) == 0))
                if id > 0 and (width_height_constraints[id-1][0] == "height"):
                    # apply last height constraint again
                    last_con = width_height_constraints[id-1][1]
                    _outer_offset_left = _outer_offset_left.at[id, 1].set(0.5 * last_con * (jp.sign(_outer_offset_left[id, 1]) * (jp.sign(_outer_offset_left[id, 1]) != 0) + 1 * jp.sign(_outer_offset_left[id, 1]) == 0))
            elif con_type == "height":
                _outer_offset_left = _outer_offset_left.at[id, 1].set(0.5 * con * (jp.sign(_outer_offset_left[id, 1]) * (jp.sign(_outer_offset_left[id, 1]) != 0) + 1 * jp.sign(_outer_offset_left[id, 1]) == 0))
                if id > 0 and (width_height_constraints[id-1][0] == "width"):
                    # apply last width constraint again
                    last_con = width_height_constraints[id-1][1]
                    _outer_offset_left = _outer_offset_left.at[id, 0].set(0.5 * last_con * (jp.sign(_outer_offset_left[id, 0]) * (jp.sign(_outer_offset_left[id, 0]) != 0) + 1 * jp.sign(_outer_offset_left[id, 0]) == 0))
    else:
        _outer_offset_left = 0.5*tunnel_size*_outer_offset_left/jp.linalg.norm(_outer_offset_left, axis=1, ord=ord).reshape(-1, 1)
    nodes_left = nodes + _outer_offset_left

    _bwd_offset_right = jp.vstack(list(map(lambda x: x.dot(angle_to_rot_matrix(-jp.pi/2).dot(jp.array([1, 0]))), bwd_rot_matrix)))
    _fwd_offset_right = jp.vstack(list(map(lambda x: x.dot(angle_to_rot_matrix(-jp.pi/2).dot(jp.array([1, 0]))), fwd_rot_matrix)))
    _outer_offset_right = (_bwd_offset_right + _fwd_offset_right) * (_bwd_offset_right != _fwd_offset_right) + _fwd_offset_right * (_bwd_offset_right == _fwd_offset_right)
    if width_height_constraints is not None:
        for id, (con_type, con) in enumerate(width_height_constraints):
            if con_type == "width":
                _outer_offset_right = _outer_offset_right.at[id, 0].set(0.5 * con * (jp.sign(_outer_offset_right[id, 0]) * (jp.sign(_outer_offset_right[id, 0]) != 0) + 1 * jp.sign(_outer_offset_right[id, 0]) == 0))
                if id > 0 and (width_height_constraints[id-1][0] == "height"):
                    # apply last height constraint again
                    last_con = width_height_constraints[id-1][1]
                    _outer_offset_right = _outer_offset_right.at[id, 1].set(0.5 * last_con * (jp.sign(_outer_offset_right[id, 1]) * (jp.sign(_outer_offset_right[id, 1]) != 0) + 1 * jp.sign(_outer_offset_right[id, 1]) == 0))
            elif con_type == "height":
                _outer_offset_right = _outer_offset_right.at[id, 1].set(0.5 * con * (jp.sign(_outer_offset_right[id, 1]) * (jp.sign(_outer_offset_right[id, 1]) != 0) + 1 * jp.sign(_outer_offset_right[id, 1]) == 0))
                if id > 0 and (width_height_constraints[id-1][0] == "width"):
                    # apply last width constraint again
                    last_con = width_height_constraints[id-1][1]
                    _outer_offset_right = _outer_offset_right.at[id, 0].set(0.5 * last_con * (jp.sign(_outer_offset_right[id, 0]) * (jp.sign(_outer_offset_right[id, 0]) != 0) + 1 * jp.sign(_outer_offset_right[id, 0]) == 0))
    else:
        _outer_offset_right = 0.5*tunnel_size*_outer_offset_right/jp.linalg.norm(_outer_offset_right, axis=1, ord=ord).reshape(-1, 1)
    nodes_right = nodes + _outer_offset_right

    return nodes_left, nodes_right

def distance_to_tunnel(test_point, _interp_fct_left, _interp_fct_right):
    """
    Estimates(!) whether a given point is inside tunnel or not, and returns the signed distance (negative distance: outside tunnel).
    """
    # find nearest parametrization value of both boundary curves
    # # theta_closest = scipy.optimize.minimize(lambda theta: jp.linalg.norm(_interp_fct(theta) - test_point), 0.7, bounds=[(0, 1)], method="Nelder-Mead", options={"maxiter": 100000}).x.item()
    # theta_closest_left = scipy.optimize.shgo(lambda theta: jp.linalg.norm(_interp_fct_left(theta) - test_point), bounds=[(0, 1)]).x.item()
    # theta_closest_right = scipy.optimize.shgo(lambda theta: jp.linalg.norm(_interp_fct_right(theta) - test_point), bounds=[(0, 1)]).x.item()
    theta_closest_left = jaxopt.LBFGSB(fun=lambda theta: jp.linalg.norm(_interp_fct_left(jp.array(theta)) - jp.array(test_point))).run(0.5, bounds=(0, 1))[0]
    theta_closest_right = jaxopt.LBFGSB(fun=lambda theta: jp.linalg.norm(_interp_fct_right(jp.array(theta)) - jp.array(test_point))).run(0.5, bounds=(0, 1))[0]
    left_bound_closest, right_bound_closest = _interp_fct_left(theta_closest_left), _interp_fct_right(theta_closest_right)
    left_vector = left_bound_closest - test_point
    right_vector = right_bound_closest - test_point
    inside_tunnel = jp.dot(left_bound_closest - test_point, right_bound_closest - test_point) < 0
    tunnel_distance = jp.minimum(jp.linalg.norm(left_vector), jp.linalg.norm(right_vector)) * (-1)**(~inside_tunnel)
    return tunnel_distance, theta_closest_left, theta_closest_right
import jax
import jax.numpy as jp
import jax._src.scipy.optimize.bfgs
import interpax
import jaxopt

def cross2d(x, y):
    return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

def heading_angle(parametrization_fct, theta, delta=1e-3, angle_change=False, rel_vec=jp.array([1, 0])
):
    if angle_change:
        # compute change in heading angle
        _fwd_delta, _bwd_delta = parametrization_fct(theta + delta) - parametrization_fct(theta), parametrization_fct(theta) - parametrization_fct(jp.maximum(theta - delta, 0))
        angle = jp.arccos(jp.dot(_fwd_delta, -_bwd_delta)/(jp.linalg.norm(_fwd_delta)*jp.linalg.norm(_bwd_delta)))
        return angle
    else:
        # compute heading angle relative to horizontal axis (x-axis in these internal computations)
        _fwd_delta, _bwd_delta = parametrization_fct(theta + delta) - parametrization_fct(theta), parametrization_fct(theta) - parametrization_fct(jp.maximum(theta - delta, 0))
        _bwd_delta = jp.where(theta.reshape(-1, 1) == 0, _fwd_delta, _bwd_delta)
        # fwd_angle = jp.arccos(jp.dot(_fwd_delta, rel_vec)/(jp.linalg.norm(_fwd_delta)*jp.linalg.norm(rel_vec))) % jp.pi
        # bwd_angle = jp.arccos(jp.dot(-_bwd_delta, rel_vec)/(jp.linalg.norm(-_bwd_delta)*jp.linalg.norm(rel_vec))) % jp.pi
        fwd_angle = (jp.arctan2(-(cross2d(_fwd_delta, rel_vec)), jp.dot(_fwd_delta, rel_vec)) + jp.pi) % (2*jp.pi) - jp.pi
        bwd_angle = (jp.arctan2(-(cross2d(_bwd_delta, rel_vec)), jp.dot(_bwd_delta, rel_vec)) + jp.pi) % (2*jp.pi) - jp.pi
        return fwd_angle, bwd_angle
    
def combine_fwd_bwd_angle(fwd_angle, bwd_angle, theta=None, delta=1e-3):
    """
    Stacks two vectors of angles computed via finite differences.
    NOTE: This function assumes that fwd_angle and bwd_angle correspond to forward differences at theta and (theta-delta), respectively.
    """
    if theta is None:
        theta = jp.linspace(0, 1, len(fwd_angle))
    theta_angle = jp.ravel(jp.vstack((theta - delta, theta)), order="F")[1:]
    angle = jp.ravel(jp.vstack((bwd_angle, fwd_angle)), order="F")[1:]
    return theta_angle, angle
    
def angle_to_rot_matrix(angle):
    return jp.array([[jp.cos(angle), -jp.sin(angle)], [jp.sin(angle), jp.cos(angle)]])

def heading_rotation_matrix(parametrization_fct, theta, delta=1e-3):
    # fwd_angle, bwd_angle = jax.vmap(functools.partial(heading_angle, parametrization_fct=parametrization_fct, delta=delta, angle_change=False))(theta=theta)
    fwd_angle, bwd_angle = heading_angle(theta=theta, parametrization_fct=parametrization_fct, delta=delta, angle_change=False)
    fwd_rot_matrix = jax.vmap(angle_to_rot_matrix)(fwd_angle)
    bwd_rot_matrix = jax.vmap(angle_to_rot_matrix)(bwd_angle)
    return fwd_rot_matrix, bwd_rot_matrix

def tunnel_from_nodes(nodes, tunnel_size=None, ord=jp.inf, width_height_constraints=None):
    assert tunnel_size is not None or width_height_constraints is not None
    
    if (isinstance(tunnel_size, jax.Array) and tunnel_size.size > 1) or isinstance(tunnel_size, list):
        assert len(tunnel_size) == len(nodes), "If multiple tunnel sizes are provided, the length of 'tunnel_size' needs to match the length of 'nodes'."
        tunnel_size = jp.array(tunnel_size).reshape(-1, 1)
    elif tunnel_size is not None:
        tunnel_size = jp.repeat(tunnel_size, len(nodes)).reshape(-1, 1)

    theta = jp.linspace(0, 1, len(nodes))
    _interp_fct = interpax.PchipInterpolator(theta, jp.array(nodes), check=False) #, k=1)

    # fwd_angle, bwd_angle = jax.vmap(functools.partial(heading_angle, parametrization_fct=_interp_fct, angle_change=False))(theta=theta)
    fwd_angle, bwd_angle = heading_angle(theta=theta, parametrization_fct=_interp_fct, angle_change=False)
    fwd_rot_matrix = jax.vmap(angle_to_rot_matrix)(fwd_angle)
    bwd_rot_matrix = jax.vmap(angle_to_rot_matrix)(bwd_angle)
    theta_angle, angle = combine_fwd_bwd_angle(fwd_angle=fwd_angle, bwd_angle=bwd_angle, theta=theta)

    _bwd_offset_left = jp.vstack(list(map(lambda x: x.dot(angle_to_rot_matrix(jp.pi/2).dot(jp.array([1, 0]))), bwd_rot_matrix)))
    _fwd_offset_left = jp.vstack(list(map(lambda x: x.dot(angle_to_rot_matrix(jp.pi/2).dot(jp.array([1, 0]))), fwd_rot_matrix)))

    _outer_offset_left = (_bwd_offset_left + _fwd_offset_left) * (_bwd_offset_left != _fwd_offset_left) + _fwd_offset_left * (_bwd_offset_left == _fwd_offset_left)
    if width_height_constraints is not None:
        for id, (con_type, con) in enumerate(width_height_constraints):
            if con_type == "width":
                _outer_offset_left = _outer_offset_left.at[id, 0].set(0.5 * con * (jp.sign(_outer_offset_left.at[id, 0].get()) + (jp.sign(_outer_offset_left.at[id, 0].get()) == 0)))
                if id > 0 and (width_height_constraints[id-1][0] == "height"):
                    # apply last height constraint again
                    last_con = width_height_constraints[id-1][1]
                    _outer_offset_left = _outer_offset_left.at[id, 1].set(0.5 * last_con * (jp.sign(_outer_offset_left.at[id, 1].get()) + (jp.sign(_outer_offset_left.at[id, 1].get()) == 0)))
            elif con_type == "height":
                _outer_offset_left = _outer_offset_left.at[id, 1].set(0.5 * con * (jp.sign(_outer_offset_left.at[id, 1].get()) + (jp.sign(_outer_offset_left.at[id, 1].get()) == 0)))
                if id > 0 and (width_height_constraints[id-1][0] == "width"):
                    # apply last width constraint again
                    last_con = width_height_constraints[id-1][1]
                    _outer_offset_left = _outer_offset_left.at[id, 0].set(0.5 * last_con * (jp.sign(_outer_offset_left.at[id, 0].get()) + (jp.sign(_outer_offset_left.at[id, 0].get()) == 0)))
    else:
        _outer_offset_left = 0.5*tunnel_size*_outer_offset_left/jp.linalg.norm(_outer_offset_left, axis=1, ord=ord).reshape(-1, 1)
    nodes_left = nodes + _outer_offset_left

    _bwd_offset_right = jp.vstack(list(map(lambda x: x.dot(angle_to_rot_matrix(-jp.pi/2).dot(jp.array([1, 0]))), bwd_rot_matrix)))
    _fwd_offset_right = jp.vstack(list(map(lambda x: x.dot(angle_to_rot_matrix(-jp.pi/2).dot(jp.array([1, 0]))), fwd_rot_matrix)))
    _outer_offset_right = (_bwd_offset_right + _fwd_offset_right) * (_bwd_offset_right != _fwd_offset_right) + _fwd_offset_right * (_bwd_offset_right == _fwd_offset_right)
    if width_height_constraints is not None:
        for id, (con_type, con) in enumerate(width_height_constraints):
            if con_type == "width":
                _outer_offset_right = _outer_offset_right.at[id, 0].set(0.5 * con * (jp.sign(_outer_offset_right.at[id, 0].get()) + (jp.sign(_outer_offset_right.at[id, 0].get()) == 0)))
                if id > 0 and (width_height_constraints[id-1][0] == "height"):
                    # apply last height constraint again
                    last_con = width_height_constraints[id-1][1]
                    _outer_offset_right = _outer_offset_right.at[id, 1].set(0.5 * last_con * (jp.sign(_outer_offset_right.at[id, 1].get()) + (jp.sign(_outer_offset_right.at[id, 1].get()) == 0)))
            elif con_type == "height":
                _outer_offset_right = _outer_offset_right.at[id, 1].set(0.5 * con * (jp.sign(_outer_offset_right.at[id, 1].get()) + (jp.sign(_outer_offset_right.at[id, 1].get()) == 0)))
                if id > 0 and (width_height_constraints[id-1][0] == "width"):
                    # apply last width constraint again
                    last_con = width_height_constraints[id-1][1]
                    _outer_offset_right = _outer_offset_right.at[id, 0].set(0.5 * last_con * (jp.sign(_outer_offset_right.at[id, 0].get()) + (jp.sign(_outer_offset_right.at[id, 0].get()) == 0)))
    else:
        _outer_offset_right = 0.5*tunnel_size*_outer_offset_right/jp.linalg.norm(_outer_offset_right, axis=1, ord=ord).reshape(-1, 1)
    nodes_right = nodes + _outer_offset_right

    return nodes_left, nodes_right, theta_angle, angle

def distance_to_tunnel(test_point, _interp_fct_left=None, _interp_fct_right=None, buffer_theta=None, buffer_nodes_left=None, buffer_nodes_right=None, theta_init=2/3, theta_min=0., theta_max=1.):
    """
    Estimates(!) whether a given point is inside tunnel or not, and returns the signed distance (negative distance: outside tunnel).
    NOTES:
    - To avoid starting the optimization at a discontinuity, theta_init should NOT be set to 1/(len(nodes)-1), 2/(len(nodes)-1), ..., (len(nodes)-2)/(len(nodes)-1).
    - To perform a warm start, the theta identified at the last step can be passed as theta_init.
    """
    # find nearest parametrization value of both boundary curves
    # # theta_closest = scipy.optimize.minimize(lambda theta: jp.linalg.norm(_interp_fct(theta) - test_point), 0.7, bounds=[(0, 1)], method="Nelder-Mead", options={"maxiter": 100000}).x.item()
    # theta_closest_left = scipy.optimize.shgo(lambda theta: jp.linalg.norm(_interp_fct_left(theta) - test_point), bounds=[(0, 1)]).x.item()
    # theta_closest_right = scipy.optimize.shgo(lambda theta: jp.linalg.norm(_interp_fct_right(theta) - test_point), bounds=[(0, 1)]).x.item()
    # theta_closest_left = jaxopt.GradientDescent(fun=lambda theta: jp.linalg.norm(_interp_fct_left(jp.array(theta)) - jp.array(test_point))).run(jp.array(theta_init)).params
    # theta_closest_right = jaxopt.GradientDescent(fun=lambda theta: jp.linalg.norm(_interp_fct_right(jp.array(theta)) - jp.array(test_point))).run(jp.array(theta_init)).params
    # theta_closest_left = jaxopt.ProjectedGradient(fun=lambda theta: jp.linalg.norm(_interp_fct_left(jp.array(theta)) - jp.array(test_point)), projection=jaxopt.projection.projection_box).run(jp.array(theta_init), hyperparams_proj=(0, 1)).params
    # theta_closest_right = jaxopt.ProjectedGradient(fun=lambda theta: jp.linalg.norm(_interp_fct_right(jp.array(theta)) - jp.array(test_point)), projection=jaxopt.projection.projection_box).run(jp.array(theta_init), hyperparams_proj=(0, 1)).params
    # theta_closest_left = jax._src.scipy.optimize.bfgs.minimize_bfgs(x0=jp.array([theta_init]), fun=lambda theta: jp.linalg.norm(_interp_fct_left(jp.array(theta)) - jp.array(test_point))).x_k.item()
    # theta_closest_right = jax._src.scipy.optimize.bfgs.minimize_bfgs(x0=jp.array([theta_init]), fun=lambda theta: jp.linalg.norm(_interp_fct_right(jp.array(theta)) - jp.array(test_point))).x_k.item()
    
    if buffer_nodes_left is None or buffer_nodes_right is None:
        assert _interp_fct_left is not None and _interp_fct_left is not None
        buffer_size = 101
        buffer_theta = jp.linspace(0, 1, buffer_size)
        buffer_nodes_left = _interp_fct_left(buffer_theta)
        buffer_nodes_right = _interp_fct_right(buffer_theta)

    # "clip" range of possible thetas by setting all node positions that are inaccessible (theta >= theta_max) to [jp.inf, jp.inf]
    # _id_max = jp.round(theta_max*(len(buffer_theta)-1)).astype(jp.int32)
    theta_mask = (jp.logical_or(buffer_theta < theta_min, buffer_theta > theta_max)).reshape(-1, 1) * jp.array([1e+8, 1e+8])
    buffer_nodes_left = buffer_nodes_left + theta_mask
    buffer_nodes_right = buffer_nodes_right + theta_mask
    
    buffer_id_closest_left = jp.argmin(jp.linalg.norm(buffer_nodes_left - jp.array(test_point), axis=1))
    buffer_id_closest_right = jp.argmin(jp.linalg.norm(buffer_nodes_right - jp.array(test_point), axis=1))
    theta_closest_left, theta_closest_right = buffer_theta[buffer_id_closest_left], buffer_theta[buffer_id_closest_right]

    # left_bound_closest, right_bound_closest = _interp_fct_left(theta_closest_left), _interp_fct_right(theta_closest_right)
    left_bound_closest, right_bound_closest = buffer_nodes_left[buffer_id_closest_left], buffer_nodes_right[buffer_id_closest_right]
    left_vector = left_bound_closest - test_point
    right_vector = right_bound_closest - test_point
    inside_tunnel = jp.dot(left_bound_closest - test_point, right_bound_closest - test_point) < 0
    # tunnel_distance = jp.minimum(jp.linalg.norm(left_vector), jp.linalg.norm(right_vector)) * (-1)**(~inside_tunnel)
    tunnel_distance_left = jp.linalg.norm(left_vector) * (-1)**(~inside_tunnel)
    tunnel_distance_right = jp.linalg.norm(right_vector) * (-1)**(~inside_tunnel)
    return tunnel_distance_left, tunnel_distance_right, theta_closest_left, theta_closest_right, left_bound_closest, right_bound_closest

def find_body_by_name(spec, name):
    for body in spec.worldbody.bodies:
        if body.name == name:
            return body
    return None

def spiral_r(theta, w):
    return jp.pow(theta+w, 3)

def spiral_r_middle(theta, w):
    """Trying to get the middle of the spiral"""
    first = spiral_r(theta, w)
    second = spiral_r(theta, w - 2*jp.pi)
    return (first + second) / 2

def to_cartesian(theta, r):
    x = r * jp.cos(theta)
    y = r * jp.sin(theta)
    return x, y

def normalise_to_max(x, y, maximum):
    max_x = jp.max(jp.abs(x))
    max_y = jp.max(jp.abs(y))
    greater = jp.maximum(max_x, max_y)
    multiplier = maximum / greater
    x *= multiplier
    y *= multiplier
    return x, y, multiplier

def normalise(x, y, multiplier):
    x *= multiplier
    y *= multiplier
    return x, y

def rotate(x, y, angle):
    return x * jp.cos(angle) - y * jp.sin(angle), x * jp.sin(angle) + y * jp.cos(angle)

def _check_rollout_for_undesired_loops(rollout):
    """Returns true if no weird loop behaviour as observed for some evals with circle_0 policies has been found."""

    _eepos = jp.array([r.info["fingertip_past"][1:] for r in rollout if r.metrics["completed_phase_0"] and (r.metrics["percentage_achieved"] >= 0.05) and (r.metrics["percentage_achieved"] <= 0.95)])
    # _fwd_deltas = _eepos[1:] - _eepos[:-1]
    # fwd_angle = np.array([(jp.arctan2(-(cross2d(_fwd_delta, rel_vec)), jp.dot(_fwd_delta, rel_vec)) + jp.pi) % (2*jp.pi) - jp.pi for _fwd_delta in _fwd_deltas])
    
    _fwd_deltas = _eepos[2:] - _eepos[1:-1]
    _bwd_deltas = _eepos[1:-1] - _eepos[:-2]
    fwd_angle = jp.array([(jp.arctan2(-(cross2d(_fwd_delta, _bwd_delta)), jp.dot(_fwd_delta, _bwd_delta)) + jp.pi) % (2*jp.pi) - jp.pi for _fwd_delta, _bwd_delta in zip(_fwd_deltas, _bwd_deltas)])

    fwd_angle_integrated = jp.trapezoid(fwd_angle)
    no_undesired_loops = fwd_angle_integrated < 1.5 * (2*jp.pi)  #more than 1.5 full circles are suspicious...
    
    return no_undesired_loops

def _check_stacked_states_for_undesired_loops(stacked_states):
    """Returns true if no weird loop behaviour as observed for some evals with circle_0 policies has been found."""

    # _eepos = jp.array([r.info["fingertip_past"][1:] for r in rollout if r.metrics["completed_phase_0"] and (r.metrics["percentage_achieved"] >= 0.05) and (r.metrics["percentage_achieved"] <= 0.95)])
    batch_size, max_length = stacked_states.info["fingertip_past"][..., 1:].shape[:2]
    _eepos = stacked_states.info["fingertip_past"][..., 1:] * (stacked_states.metrics["completed_phase_0"] * (stacked_states.metrics["percentage_achieved"] >= 0.05) * (stacked_states.metrics["percentage_achieved"] <= 0.95)).reshape(batch_size, max_length, 1)
    # _fwd_deltas = _eepos[1:] - _eepos[:-1]
    # fwd_angle = np.array([(jp.arctan2(-(cross2d(_fwd_delta, rel_vec)), jp.dot(_fwd_delta, rel_vec)) + jp.pi) % (2*jp.pi) - jp.pi for _fwd_delta in _fwd_deltas])
    
    _fwd_deltas = _eepos[:, 2:] - _eepos[:, 1:-1]
    _bwd_deltas = _eepos[:, 1:-1] - _eepos[:, :-2]
    # print(_fwd_deltas.shape, _bwd_deltas.shape, _eepos.shape, batch_size, max_length)
    fwd_angle = (jp.arctan2(-(cross2d(_fwd_deltas, _bwd_deltas)), jp.sum(_fwd_deltas * _bwd_deltas, axis=-1)) + jp.pi) % (2*jp.pi) - jp.pi

    fwd_angle_integrated = jp.trapezoid(fwd_angle, axis=-1)
    undesired_loops = fwd_angle_integrated >= 1.5 * (2*jp.pi)  #more than 1.5 full circles are suspicious...
    
    return undesired_loops
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import pandas as pd
from collections import defaultdict

def preprocess_steering_law_rollouts(movement_times, rollout_states, task, average_r2 = True):
    movement_times = np.array(movement_times)

    if task in ("circle_0",):
        Ws = np.array([np.abs(r.info["tunnel_nodes_left"][0,1] - r.info["tunnel_nodes_right"][0,1]) for r in rollout_states])
        Rs = np.array([np.abs(0.5*(r.info["tunnel_nodes_left"][:,1].max() - r.info["tunnel_nodes_right"][:,1].min())) for r in rollout_states])
        Ds = Rs * (2 * np.pi)
        Xs = np.array([r.info["tunnel_nodes"][:,0] for r in rollout_states])
        Ys = np.array([r.info["tunnel_nodes"][:,1] for r in rollout_states])
        IDs = (Ds / Ws).reshape(-1, 1)
        sl_data = {"ID": IDs, "MT_ref": movement_times,
                    "D": Ds, "W": Ws, "R": Rs, "X": Xs, "Y": Ys}
    elif task in ('menu_0', 'menu_1', 'menu_2'):
        Ds = np.array([np.abs(-r.info["tunnel_nodes_left"][1:,1] + r.info["tunnel_nodes_right"][:-1,1]) for r in rollout_states])
        Ws = np.array([np.abs(r.info["tunnel_nodes_left"][:-1,0] - r.info["tunnel_nodes_right"][1:,0]) for r in rollout_states])
        IDs = np.stack([
            Ds[:, i] / Ws[:, i] if (i % 2 == 0) else Ws[:, i] / Ds[:, i]
            for i in range(Ds.shape[1])
        ], axis=1).sum(axis=1).reshape(-1, 1) if len(Ds) > 0 else np.array([])
        Xs = np.array([r.info["tunnel_nodes"][:,0] for r in rollout_states])
        Ys = np.array([r.info["tunnel_nodes"][:,1] for r in rollout_states])
        sl_data = {"ID": IDs, "MT_ref": movement_times,
                    "D": Ds, "W": Ws, "X": Xs, "Y": Ys}
    elif task in ('spiral_0'):
        Ds = np.array([np.linalg.vector_norm(np.abs(r.info["tunnel_nodes"][1:] - r.info["tunnel_nodes"][:-1]), axis=-1) for r in rollout_states])
        Ws = np.array([np.linalg.vector_norm(np.abs(r.info["tunnel_nodes_left"] - r.info["tunnel_nodes_right"]), axis=-1)[1:] for r in rollout_states])
        Rs = np.array([0.5*(r.info["tunnel_extras"]["r_inner"] + r.info["tunnel_extras"]["r_outer"])[1:] for r in rollout_states])
        Xs = np.array([r.info["tunnel_nodes"][:,0] for r in rollout_states])
        Ys = np.array([r.info["tunnel_nodes"][:,1] for r in rollout_states])
        IDs = np.sum(Ds / Ws, axis=-1).reshape(-1, 1)
        sl_data = {"ID": IDs, "MT_ref": movement_times,
                    "D": Ds, "W": Ws, "R": Rs, "X": Xs, "Y": Ys}
    elif task in ('sinusoidal_0',):
        Ds = np.array([np.linalg.vector_norm(np.abs(r.info["tunnel_nodes"][1:] - r.info["tunnel_nodes"][:-1]), axis=-1) for r in rollout_states])
        Ws = np.array([np.linalg.vector_norm(np.abs(r.info["tunnel_nodes_left"] - r.info["tunnel_nodes_right"]), axis=-1)[1:] for r in rollout_states])
        IDs = np.sum(Ds / Ws, axis=-1).reshape(-1, 1)
        Xs = np.array([r.info["tunnel_nodes"][:,0] for r in rollout_states])
        Ys = np.array([r.info["tunnel_nodes"][:,1] for r in rollout_states])
        sl_data = {"ID": IDs, "MT_ref": movement_times,
                    "D": Ds, "W": Ws, "X": Xs, "Y": Ys}
    elif task in ('rectangle_0',):
        Ds = np.array([np.abs(r.info["tunnel_nodes"][-1, 0] - r.info["tunnel_nodes"][0, 0]) for r in rollout_states])
        Ws = np.array([np.abs(r.info["tunnel_nodes_right"][0, 1] - r.info["tunnel_nodes_left"][0, 1]) for r in rollout_states])
        Xs = np.array([r.info["tunnel_nodes"][:,0] for r in rollout_states])
        Ys = np.array([r.info["tunnel_nodes"][:,1] for r in rollout_states])
        IDs = (Ds / Ws).reshape(-1, 1)
        sl_data = {"ID": IDs, "MT_ref": movement_times,
                    "D": Ds, "W": Ws, "X": Xs, "Y": Ys}
    elif task in ('varying_width',):
        IDs = np.array([np.abs(r.info["tunnel_extras"]["ID"]) for r in rollout_states]).reshape(-1, 1)
        Xs = np.array([r.info["tunnel_nodes"][:,0] for r in rollout_states])
        Ys = np.array([r.info["tunnel_nodes"][:,1] for r in rollout_states])
        sl_data = {"ID": IDs, "MT_ref": movement_times, "X": Xs, "Y": Ys}
    else:
        raise NotImplementedError()
    
    if average_r2:
        return average_movement_times_per_path(sl_data, outlier_std=3)
    else:
        return sl_data

def average_movement_times_per_path(sl_data, outlier_std=None, outlier_proportiontocut=None):
    groups = defaultdict(list)
    n_episodes = sl_data["X"].shape[0]

    for i in range(n_episodes):
        key = (tuple(sl_data["X"][i]), tuple(sl_data["Y"][i]))
        groups[key].append(i)

    results = {k: [] for k in sl_data.keys()}
    results["ID_means"] = []
    results["MT_ref_means"] = []

    for _, idxs in groups.items():
        idx = idxs[0]
        for k in sl_data.keys():
            if k not in ("MT_ref", "ID"):
                results[k].append(np.array(sl_data[k][idx]))
        
        mt_vals = np.array(sl_data["MT_ref"])[idxs]
        if mt_vals.size > 2:
            if outlier_std is not None:
                z_scores = scipy.stats.zscore(mt_vals)
                z_scores[np.isnan(z_scores)] = 0
                mt_vals_wo_outliers = mt_vals[abs(z_scores) <= outlier_std]
            else:
                mt_vals_wo_outliers = mt_vals.copy()
            if outlier_proportiontocut is not None:
                mt_vals_wo_outliers = scipy.stats.trimboth(mt_vals_wo_outliers, proportiontocut=outlier_proportiontocut)
        else:
            mt_vals_wo_outliers = mt_vals

        mt_mean = np.mean(mt_vals_wo_outliers)
        results["MT_ref_means"].append(mt_mean)
        results["MT_ref"].append(np.array(sl_data["MT_ref"])[idxs])

        id_vals = sl_data["ID"][idxs]
        if id_vals.size > 2:
            if outlier_std is not None:
                z_scores = scipy.stats.zscore(mt_vals)
                z_scores[np.isnan(z_scores)] = 0
                id_vals_wo_outliers = id_vals[abs(z_scores) <= outlier_std]
            else:
                id_vals_wo_outliers = id_vals.copy()
            if outlier_proportiontocut is not None:
                id_vals_wo_outliers = scipy.stats.trimboth(id_vals_wo_outliers, proportiontocut=outlier_proportiontocut)  #TODO: fix (only works if all entries of id_vals_wo_outliers are the same)!
        else:
            id_vals_wo_outliers = id_vals
        id_mean = np.mean(id_vals_wo_outliers)
        results["ID_means"].append(id_mean)
        results["ID"].append(sl_data["ID"][idxs])

    results["ID_means"] = np.array(results["ID_means"]).reshape(-1, 1)

    return results

def calculate_curvature(Xs, Ys):
    # Approximate 1st Derivative
    dx = np.diff(Xs, axis=-1)     # (N, M-1)
    dy = np.diff(Ys, axis=-1)     # (N, M-1)
    # Approximate 2nd Derivative
    ddx = np.diff(Xs, n=2, axis=-1)  # (N, M-2)
    ddy = np.diff(Ys, n=2, axis=-1)  # (N, M-2)
    ds = np.sqrt(dx**2 + dy**2)  # (N, M-1)
    #same length
    dx = dx[:, :-1]
    dy = dy[:, :-1]
    ds = ds[:, :-1]

    # curvature
    kappa = np.abs(dx * ddy - dy * ddx) / (ds**3)
    return kappa, ds

def calculate_steering_laws(movement_times, rollout_states, task, plot_data=False, average_r2=True):
    sl_data = preprocess_steering_law_rollouts(movement_times=movement_times, rollout_states=rollout_states, task=task, average_r2=average_r2)
    a,b,r2,sl_data0 = calculate_original_steering_law(sl_data.copy(), average_r2=average_r2)
    plot_metrics = {}
    if plot_data and isinstance(sl_data0, dict):
        plot_metrics['plot_original'] =  sl_data0.copy()
        plot_metrics['plot_original']['r2'] = r2

    if np.isnan(r2):
        return {}
    else:
        if task in ('circle_0',):
            r2_nancel,sl_data1 = calculate_nancel_steering_law(sl_data.copy(), task, average_r2=average_r2)
            r2_yamanaka, sl_data2 = calculate_yamanaka_steering_law(sl_data.copy(), average_r2=average_r2)
            r2_liu, sl_data3 = calculate_liu_steering_law(sl_data.copy(), average_r2=average_r2)
            metrics = {'SL/r2': r2, 'SL/b': b, 'SL/len(ID_means)': len(sl_data0['ID_means']),'SL/r2_nancel': r2_nancel, 'SL/r2_yamanaka': r2_yamanaka, 'SL/r2_liu': r2_liu}
            if plot_data:
                if isinstance(sl_data1, dict):
                    plot_metrics['plot_nancel'] = sl_data1.copy()
                    plot_metrics['plot_nancel']['r2'] = r2_nancel
                if isinstance(sl_data2, dict):
                    plot_metrics['plot_yamanaka'] = sl_data2.copy()
                    plot_metrics['plot_yamanaka']['r2'] = r2_yamanaka
                if isinstance(sl_data3, dict):
                    plot_metrics['plot_liu'] = sl_data3.copy()
                    plot_metrics['plot_liu']['r2'] = r2_liu
        elif task in ('menu_0', 'menu_1', 'menu_2'):
            r2_ahlstroem, sl_data1 = calculate_ahlstroem_steering_law(sl_data.copy(), average_r2=average_r2)
            metrics = {'SL/r2': r2, 'SL/b': b, 'SL/len(ID_means)': len(sl_data0['ID_means']), 'SL/r2_ahlstroem': r2_ahlstroem}
            if plot_data:
                if isinstance(sl_data1, dict):
                    plot_metrics['plot_ahlstroem'] = sl_data1.copy()
                    plot_metrics['plot_ahlstroem']['r2'] = r2_ahlstroem
        elif task in ('spiral_0'):
            r2_nancel,sl_data1 = calculate_nancel_steering_law(sl_data.copy(), task, average_r2=average_r2)
            #r2_chen,sl_data2 = calculate_chen_steering_law(sl_data.copy())
            metrics = {'SL/r2': r2, 'SL/b': b, 'SL/len(ID_means)': len(sl_data0['ID_means']),'SL/r2_nancel': r2_nancel}#,'SL/r2_chen': r2_chen}
            if plot_data:
                if isinstance(sl_data1, dict):
                    plot_metrics['plot_nancel'] = sl_data1.copy()
                    plot_metrics['plot_nancel']['r2'] = r2_nancel
        elif task in ('sinusoidal_0'):
            r2_nancel,sl_data1 = calculate_nancel_steering_law(sl_data.copy(), task, average_r2=average_r2)
            r2_chen,sl_data2 = calculate_chen_steering_law(sl_data.copy(), average_r2=average_r2)
            metrics = {'SL/r2': r2, 'SL/b': b, 'SL/len(ID_means)': len(sl_data0['ID_means']),'SL/r2_nancel': r2_nancel,'SL/r2_chen': r2_chen}
            if plot_data:
                if isinstance(sl_data1, dict):
                    plot_metrics['plot_nancel'] = sl_data1.copy()
                    plot_metrics['plot_nancel']['r2'] = r2_nancel
                if isinstance(sl_data2, dict):
                    plot_metrics['plot_chen'] = sl_data2.copy()
                    plot_metrics['plot_chen']['r2'] = r2_chen

        else:
            metrics = {'SL/r2': r2, 'SL/b': b, 'SL/len(ID_means)': len(sl_data0['ID_means'])}
        metrics.update(plot_metrics)
        return metrics

def calculate_chen_steering_law(sl_data, average_r2=True): 

    if average_r2:
        MTs = sl_data["MT_ref_means"]
    else:
        MTs = sl_data["MT_ref"]

    Ds = np.sum(sl_data["D"], axis=-1)
    kappa, ds = calculate_curvature(sl_data["X"], sl_data["Y"])
    Ks = np.sum(kappa * ds, axis=-1)

    def residuals(params):
        a, b, c, d = params
        x1 = Ds
        x2 = np.log2(Ks+1)
        x3 = Ds*Ks
        MT_pred = a + b*x1 + c*x2 + d*x3
        return (MT_pred - MTs).ravel() 

    x0 = [np.mean(MTs), 1.0, 0.0, 0.0]

    res = least_squares(residuals, x0)

    a, b, c, d = res.x
    x1 = Ds
    x2 = np.log2(Ks+1)
    x3 = Ds*Ks
    MT_pred = a + b*x1 + c*x2 + d*x3
    x_values = x1 + c/b * x2 + d/b * x3

    r2 = r2_score(MTs, MT_pred)

    sl_data.update({"MT_pred": MT_pred, "x_values": x_values})
    return r2, sl_data
        
def calculate_nancel_steering_law(sl_data, task='circle_0', average_r2=True):
    if average_r2:
        MTs = np.array(sl_data["MT_ref_means"]).reshape(1, -1)
    else:
        MTs = sl_data["MT_ref"]

    if task == 'circle_0':
        IDs = ((np.power(sl_data['R'], 1/3) / sl_data['W'])*sl_data['D']).reshape(-1, 1)
    elif task in ('spiral_0', 'sinusoidal_0'):
        dx = np.diff(sl_data["X"], axis=-1)     # (N, M-1)
        dy = np.diff(sl_data["Y"], axis=-1)     # (N, M-1)
        ds = np.sqrt(dx**2 + dy**2)  # (N, M-1)
        Ws = np.array(sl_data['W'])
        if task == "spiral_0":
            Rs = sl_data['R']
            f_vals = 1 / (Ws * np.power(Rs, 1/3))
        else:
            kappa, ds = calculate_curvature(sl_data["X"], sl_data["Y"])
            Rs = 1/kappa
            # To Do: W überarbeiten
            f_vals = 1 / (Ws[:-1] * np.power(Rs, 1/3))
        IDs = np.sum(f_vals * ds, axis=-1).reshape(1, -1)
    else:
        print(f"Not implemented for this task {task}")
        return np.nan, sl_data
    
    IDs = IDs.ravel().reshape(-1, 1)
    MTs = MTs.ravel()
    a, b, r2, y_pred = fit_model(IDs, MTs)
    print(f"R^2: {r2}, a,b: {a},{b}")

    sl_data.update({"MT_pred": y_pred, "x_values": IDs})
    return r2,sl_data

def calculate_yamanaka_steering_law(sl_data, average_r2=True):
    if average_r2:
        MTs = sl_data["MT_ref_means"]
    else:
        MTs = sl_data["MT_ref"]

    Ds = np.array(sl_data['D'])
    Ws = np.array(sl_data['W'])
    Rs = np.array(sl_data['R'])


    def residuals(params):
        a, b, c, d = params
        denom = Ws + c * (1.0 / Rs) + d * Ws * (1.0 / Rs)
        MT_pred = a + b * (Ds / denom)
        return (MT_pred - MTs).ravel()

    x0 = [np.mean(MTs), 1.0, 0.0, 0.0]

    res = least_squares(residuals, x0)

    a, b, c, d = res.x
    IDs = (Ds / (Ws + c*(1.0/Rs) + d*Ws*(1.0/Rs)))
    MT_pred = a + b * IDs

    r2 = r2_score(MTs, MT_pred)

    sl_data.update({"MT_pred": MT_pred, "x_values": IDs})
    return r2, sl_data

def calculate_liu_steering_law(sl_data, average_r2=True):
    if average_r2:
        MTs = sl_data["MT_ref_means"]
    else:
        MTs = sl_data["MT_ref"]

    Ds = np.array(sl_data['D'])
    Ws = np.array(sl_data['W'])
    Rs = np.array(sl_data['R'])

    y = np.log(MTs)

    def residuals(params):
        a, b, c, d = params
        x1 = np.log(Ds / Ws)
        x2 = 1.0 / Rs
        x3 = (1.0 / Rs) * np.log(Ds / Ws)
        y_pred = a + b*x1 + c*x2 + d*x3
        return (y_pred - y).ravel() 

    x0 = [np.mean(y), 1.0, 0.0, 0.0]

    res = least_squares(residuals, x0)

    a, b, c, d = res.x

    x1 = np.log(Ds / Ws)
    x2 = 1.0 / Rs
    x3 = (1.0 / Rs) * np.log(Ds / Ws)
    y_pred = a + b*x1 + c*x2 + d*x3

    r2 = r2_score(y, y_pred)

    x_values = x1 + c/b * x2 + d/b * x3

    sl_data.update({"MT_pred": np.exp(y_pred), "x_values": x_values, "y_pred": y_pred})

    return r2, sl_data

def calculate_ahlstroem_steering_law(sl_data, average_r2=True):
    if average_r2:
        MTs = sl_data["MT_ref_means"]
    else:
        MTs = sl_data["MT_ref"]
    
    Ds = np.array(sl_data['D'])
    Ws = np.array(sl_data['W'])

    IDs = np.stack([
            np.log2(Ds[:, i] / Ws[:, i]+1) if i % 2 == 0 else 0.5*Ws[:, i] / Ds[:, i]
            for i in range(Ds.shape[1])
        ], axis=1).sum(axis=1).reshape(-1, 1)
    a, b, r2, y_pred= fit_model(IDs, MTs)

    x_values = IDs

    sl_data.update({"MT_pred": y_pred, "x_values": x_values})

    return r2,sl_data
    
def calculate_original_steering_law(sl_data, average_r2=True):   
    if average_r2:
        MTs = sl_data["MT_ref_means"]
        IDs = sl_data["ID_means"]
    else:
        MTs = sl_data["MT_ref"]
        IDs = sl_data["ID"]

    if len(IDs) == 0 or len(MTs) == 0:
        return np.nan, np.nan, np.nan, np.nan
    
    a, b, r2, y_pred = fit_model(np.array(IDs).reshape(-1,1), np.array(MTs))

    print(f"R^2: {r2}, a,b: {a},{b}")

    sl_data.update({"MT_pred": y_pred})

    return a,b,r2,sl_data

def fit_model(IDs, MTs):
    model = LinearRegression()
    model.fit(IDs, MTs)
    a = model.intercept_
    b = model.coef_[0]
    y_pred = model.predict(IDs)
    r2 = r2_score(MTs, y_pred)
    return a, b, r2, y_pred

def plot_steering_law(sl_data, r2, law_name='Original', average_r2=True):
    
    if not average_r2:
        plt.scatter(sl_data["ID"], sl_data["MT_ref"])
        plt.plot(sl_data["ID"], sl_data["MT_pred"],  color="red")
        plt.title(f"R^2={r2:.2g} for {law_name} steering law")
    else:
        plt.scatter(sl_data["ID_means"], sl_data["MT_means_ref"])
        plt.plot(sl_data["ID_means"], sl_data["MT_pred"], "--", color="red")
        plt.title(f"Average R^2={r2:.2g} for {law_name} steering law")
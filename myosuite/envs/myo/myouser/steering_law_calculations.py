from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import pandas as pd

def preprocess_steering_law_rollouts(movement_times, rollout_states, task):
    movement_times = np.array(movement_times)

    if task in ("circle_0",):
        Ws = np.array([np.abs(r.info["tunnel_nodes_left"][0,1] - r.info["tunnel_nodes_right"][0,1]) for r in rollout_states])
        Rs = np.array([np.abs(0.5*(r.info["tunnel_nodes_left"][:,1].max() - r.info["tunnel_nodes_right"][:,1].min())) for r in rollout_states])
        Ds = Rs * (2 * np.pi)
        IDs = (Ds / Ws).reshape(-1, 1)
        sl_data = {"ID": IDs, "MT_ref": movement_times,
                    "D": Ds, "W": Ws, "R": Rs}
    elif task in ('menu_0', 'menu_1', 'menu_2'):
        Ds = np.array([np.abs(-r.info["tunnel_nodes_left"][1:,1] + r.info["tunnel_nodes_right"][:-1,1]) for r in rollout_states])
        Ws = np.array([np.abs(r.info["tunnel_nodes_left"][:-1,0] - r.info["tunnel_nodes_right"][1:,0]) for r in rollout_states])
        IDs = np.stack([
            Ds[:, i] / Ws[:, i] if (i % 2 == 0) else Ws[:, i] / Ds[:, i]
            for i in range(Ds.shape[1])
        ], axis=1).sum(axis=1).reshape(-1, 1) if len(Ds) > 0 else np.array([])
        sl_data = {"ID": IDs, "MT_ref": movement_times,
                    "D": Ds, "W": Ws}
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
        IDs = (Ds / Ws).reshape(-1, 1)
        sl_data = {"ID": IDs, "MT_ref": movement_times,
                    "D": Ds, "W": Ws}
    else:
        raise NotImplementedError()
    return sl_data

def calculate_curvature(sl_data):
    # Approximate 1st Derivative
    dx = np.diff(sl_data["X"], axis=-1)     # (N, M-1)
    dy = np.diff(sl_data["Y"], axis=-1)     # (N, M-1)
    # Approximate 2nd Derivative
    ddx = np.diff(sl_data["X"], n=2, axis=-1)  # (N, M-2)
    ddy = np.diff(sl_data["Y"], n=2, axis=-1)  # (N, M-2)
    ds = np.sqrt(dx**2 + dy**2)  # (N, M-1)
    #same length
    dx = dx[:, :-1]
    dy = dy[:, :-1]
    ds = ds[:, :-1]

    # curvature
    kappa = np.abs(dx * ddy - dy * ddx) / (ds**3)
    return kappa, ds

def calculate_steering_laws(movement_times, rollout_states, task, average_r2=True):
    sl_data = preprocess_steering_law_rollouts(movement_times=movement_times, rollout_states=rollout_states, task=task)
    a,b,r2,sl_data0 = calculate_original_steering_law(sl_data.copy(), average_r2)

    if np.isnan(r2):
        return {}
    else:
        if task in ('circle_0',):
            r2_nancel,sl_data1 = calculate_nancel_steering_law(sl_data.copy(), task, average_r2)
            r2_yamanaka, sl_data2 = calculate_yamanaka_steering_law(sl_data.copy(), average_r2)
            r2_liu, sl_data3 = calculate_liu_steering_law(sl_data.copy(), average_r2)
            metrics = {'SL/r2': r2, 'SL/b': b, 'SL/len(ID_means)': len(sl_data0['ID_means']),'SL/r2_nancel': r2_nancel, 'SL/r2_yamanaka': r2_yamanaka, 'SL/r2_liu': r2_liu}
        elif task in ('menu_0', 'menu_1', 'menu_2'):
            r2_ahlstroem, sl_data1 = calculate_ahlstroem_steering_law(sl_data.copy(), average_r2)
            metrics = {'SL/r2': r2, 'SL/b': b, 'SL/len(ID_means)': len(sl_data0['ID_means']), 'SL/r2_ahlstroem': r2_ahlstroem}
        elif task in ('spiral_0'):
            r2_nancel,sl_data1 = calculate_nancel_steering_law(sl_data.copy(), task, average_r2)
            #r2_chen,sl_data2 = calculate_steering_law_chen(sl_data.copy(), average_r2)
            metrics = {'SL/r2': r2, 'SL/b': b, 'SL/len(ID_means)': len(sl_data0['ID_means']),'SL/r2_nancel': r2_nancel}#,'SL/r2_chen': r2_chen}
        elif task in ('sinusoidal_0'):
            r2_nancel,sl_data1 = calculate_nancel_steering_law(sl_data.copy(), task, average_r2)
            r2_chen,sl_data2 = calculate_steering_law_chen(sl_data.copy(), average_r2)
            metrics = {'SL/r2': r2, 'SL/b': b, 'SL/len(ID_means)': len(sl_data0['ID_means']),'SL/r2_nancel': r2_nancel,'SL/r2_chen': r2_chen}

        else:
            metrics = {'SL/r2': r2, 'SL/b': b, 'SL/len(ID_means)': len(sl_data0['ID_means'])}
        return metrics

def calculate_steering_law_chen(sl_data, average_r2=True):
    kappa, ds = calculate_curvature(sl_data)
    Ks = np.sum(kappa * ds, axis=-1)

    sl_data['D'] = np.sum(sl_data['D'], axis=-1)
    sl_data['W'] = np.sum(sl_data['W'], axis=-1)
    # In df to enable grouping
    sl_data['K'] = Ks
    keys = ["D", "MT_ref", "K", "W"]
    sl_data_1d = {k: np.asarray(sl_data[k]).ravel() for k in keys}
    df = pd.DataFrame(sl_data_1d)
    if average_r2:
        df = df.groupby(['D', 'W', 'K'], as_index=False)['MT_ref'].mean()
        
    MTs = sl_data['MT_ref']
    Ds = sl_data['D']

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
    if average_r2:
        sl_data.update({"MT_means_ref": MTs})
    return r2, sl_data
        
def calculate_nancel_steering_law(sl_data, task='circle_0', average_r2=True):
    if task == 'circle_0':
        IDs = ((np.power(sl_data['R'], 1/3) / sl_data['W'])*sl_data['D']).reshape(-1, 1)
    elif task in ('spiral_0', 'sinusoidal_0'):
        dx = np.diff(sl_data["X"], axis=-1)     # (N, M-1)
        dy = np.diff(sl_data["Y"], axis=-1)     # (N, M-1)
        ds = np.sqrt(dx**2 + dy**2)  # (N, M-1)
        Ws = sl_data['W']
        if task == "spiral_0":
            Rs = sl_data['R']
            f_vals = 1 / (Ws * np.power(Rs, 1/3))
        else:
            kappa, ds = calculate_curvature(sl_data)
            Rs = 1/kappa
            f_vals = 1 / (Ws[:,:-1] * np.power(Rs, 1/3))
        IDs = np.sum(f_vals * ds, axis=-1)
    else:
        print(f"Not implemented for this task {task}")
        return np.nan, sl_data

    MTs = sl_data['MT_ref']
    a, b, r2, y_pred, ID_means, MT_means = fit_model(IDs, MTs, average_r2=average_r2)
    print(f"R^2: {r2}, a,b: {a},{b}")

    sl_data.update({"MT_pred": y_pred, "x_values": IDs})
    if average_r2:
        sl_data.update({"ID_means": ID_means, "MT_means_ref": MT_means, "x_values": ID_means})
    return r2,sl_data

def calculate_yamanaka_steering_law(sl_data, average_r2=True):
    keys = ["D", "W", "R", "MT_ref"]
    sl_data_1d = {k: np.asarray(sl_data[k]).ravel() for k in keys}
    df = pd.DataFrame(sl_data_1d)

    if average_r2:
        df = df.groupby(['D', 'W', 'R'], as_index=False)['MT_ref'].mean()

    MTs = df['MT_ref'].values
    Ds = df['D'].values
    Ws = df['W'].values
    Rs = df['R'].values


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
    if average_r2:
        sl_data.update({"ID_means": IDs, "MT_means_ref": MTs, "x_values": IDs})

    return r2, sl_data

def calculate_liu_steering_law(sl_data, average_r2=True):
    keys = ["D", "W", "R", "MT_ref"]
    sl_data_1d = {k: np.asarray(sl_data[k]).ravel() for k in keys}
    df = pd.DataFrame(sl_data_1d)

    if average_r2:
        df = df.groupby(['D', 'W', 'R'], as_index=False)['MT_ref'].mean()

    MTs = df['MT_ref'].values
    Ds = df['D'].values
    Ws = df['W'].values
    Rs = df['R'].values

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
    x_axis_values = np.log(y_pred)

    sl_data.update({"MT_pred": np.exp(y_pred), "ID": x1, "x_values": x_values, "x_axis_values": x_axis_values})
    if average_r2:
        sl_data.update({"ID_means": x1, "curvature_means": x2, "MT_means_ref": MTs, "x_values": x_values, "x_axis_values": x_axis_values})

    return r2, sl_data

def calculate_ahlstroem_steering_law(sl_data, average_r2=True):
    Ds = np.array(sl_data['D'])
    Ws = np.array(sl_data['W'])
    MTs = np.array(sl_data['MT_ref'])
    IDs = np.stack([
            np.log2(Ds[:, i] / Ws[:, i]+1) if i % 2 == 0 else 0.5*Ws[:, i] / Ds[:, i]
            for i in range(Ds.shape[1])
        ], axis=1).sum(axis=1).reshape(-1, 1)
    a, b, r2, y_pred, ID_means, MT_means= fit_model(IDs, MTs, average_r2=average_r2)
    print(f"R^2: {r2}, a,b: {a},{b}")
    x_values = ID_means

    sl_data.update({"MT_pred": y_pred, "ID": ID_means, "x_values": x_values})
    if average_r2:
        sl_data.update({"ID_means": ID_means, "MT_means_ref": MT_means, "x_values": x_values})
    return r2,sl_data
    
def calculate_original_steering_law(sl_data, average_r2=True):   
    IDs = sl_data["ID"]
    MTs = sl_data["MT_ref"]

    if len(IDs) == 0 or len(MTs) == 0:
        return np.nan, np.nan, np.nan, np.nan
    
    a, b, r2, y_pred, ID_means, MT_means = fit_model(IDs, MTs, average_r2=average_r2)

    print(f"R^2: {r2}, a,b: {a},{b}")

    sl_data.update({"MT_pred": y_pred})
    if average_r2:
        sl_data.update({"ID_means": ID_means, "MT_means_ref": MT_means})

    return a,b,r2,sl_data

def fit_model(IDs, MTs, average_r2):
    if average_r2:
        IDs_rounded = IDs.round(2)
        ID_means = np.sort(np.unique(IDs_rounded)).reshape(-1, 1)
        MT_means = np.array([MTs[np.argwhere(IDs_rounded.flatten() == _id)].mean() for _id in ID_means])

        model = LinearRegression()
        model.fit(ID_means, MT_means)
        a = model.intercept_
        b = model.coef_[0]
        y_pred = model.predict(ID_means)
        r2 = r2_score(MT_means, y_pred)
    else:
        model = LinearRegression()
        model.fit(IDs, MTs)
        a = model.intercept_
        b = model.coef_[0]
        y_pred = model.predict(IDs)
        r2 = r2_score(MTs, y_pred)
        ID_means = None
        MT_means = None
    return a, b, r2, y_pred, ID_means, MT_means

def plot_steering_law(sl_data, r2, average_r2=True, law_name='Original'):
    
    if not average_r2:
        plt.scatter(sl_data["ID"], sl_data["MT_ref"])
        plt.plot(sl_data["ID"], sl_data["MT_pred"],  color="red")
        plt.title(f"R^2={r2:.2g} for {law_name} steering law")
    else:
        plt.scatter(sl_data["ID_means"], sl_data["MT_means_ref"])
        plt.plot(sl_data["ID_means"], sl_data["MT_pred"], "--", color="red")
        plt.title(f"Average R^2={r2:.2g} for {law_name} steering law")
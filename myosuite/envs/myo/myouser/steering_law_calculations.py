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
        ], axis=1).sum(axis=1).reshape(-1, 1)
        sl_data = {"ID": IDs, "MT_ref": movement_times,
                    "D": Ds, "W": Ws}
    elif task in ('spiral_0',):
        Ds = np.array([np.linalg.vector_norm(np.abs(r.info["tunnel_nodes"][1:] - r.info["tunnel_nodes"][:-1]), axis=-1) for r in rollout_states])
        Ws = np.array([np.linalg.vector_norm(np.abs(r.info["tunnel_nodes_left"] - r.info["tunnel_nodes_right"]), axis=-1)[1:] for r in rollout_states])
        IDs = np.sum(Ds / Ws).reshape(-1, 1)
        sl_data = {"ID": IDs, "MT_ref": movement_times,
                    "D": Ds, "W": Ws}
    elif task in ('rectangle_0',):
        Ds = np.array([np.abs(r.info["tunnel_nodes"][-1, 0] - r.info["tunnel_nodes"][0, 0]) for r in rollout_states])
        Ws = np.array([np.abs(r.info["tunnel_nodes_right"][0, 1] - r.info["tunnel_nodes_left"][0, 1]) for r in rollout_states])
        IDs = (Ds / Ws).reshape(-1, 1)
        sl_data = {"ID": IDs, "MT_ref": movement_times,
                    "D": Ds, "W": Ws}
    else:
        raise NotImplementedError()
    return sl_data

def calculate_steering_laws(movement_times, rollout_states, task, average_r2=True):
    sl_data = preprocess_steering_law_rollouts(movement_times=movement_times, rollout_states=rollout_states, task=task)
    a,b,r2,sl_data = calculate_original_steering_law(sl_data.copy(), average_r2)

    if np.isnan(r2):
        return {'r2': r2}
    else:
        if task in ('circle_0',):
            r2_nancel,sl_data1 = calculate_nancel_steering_law(sl_data.copy(), average_r2)
            r2_yamanaka, sl_data2 = calculate_yamanaka_steering_law(sl_data.copy(), average_r2)
            r2_liu, sl_data3 = calculate_liu_steering_law(sl_data.copy(), average_r2)
            metrics = {'r2': r2, 'r2_nancel': r2_nancel, 'r2_yamanaka': r2_yamanaka, 'r2_liu': r2_liu}
        elif task in ('menu_0', 'menu_1', 'menu_2'):
            r2_ahlstroem, sl_data1 = calculate_ahlstroem_steering_law(sl_data.copy(), average_r2)
            metrics = {'r2': r2, 'r2_ahlstroem': r2_ahlstroem}
        return metrics

def calculate_nancel_steering_law(sl_data, average_r2=True):
    IDs = (np.power(sl_data['R'], 2/3) / sl_data['W']).reshape(-1, 1)
    MTs = sl_data['MT_ref']

    a, b, r2, y_pred, ID_means, MT_means = fit_model(IDs, MTs, average_r2=average_r2)
    print(f"R^2: {r2}, a,b: {a},{b}")

    sl_data.update({"MT_pred": y_pred})
    if average_r2:
        sl_data.update({"ID_means": ID_means, "MT_means_ref": MT_means})
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
    ID_means = (Ds / (Ws + c*(1.0/Rs) + d*Ws*(1.0/Rs)))
    MT_pred = a + b * ID_means

    r2 = r2_score(MTs, MT_pred)

    sl_data.update({"MT_pred": MT_pred})
    if average_r2:
        sl_data.update({"ID_means": ID_means, "MT_means_ref": MTs})

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

    sl_data.update({"MT_pred": np.exp(y_pred)})
    if average_r2:
        sl_data.update({"ID_means": x1, "curvature_means": x2, "MT_means_ref": MTs})

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

    sl_data.update({"MT_pred": y_pred})
    if average_r2:
        sl_data.update({"ID_means": ID_means, "MT_means_ref": MT_means})
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
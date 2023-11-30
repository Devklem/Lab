import streamlit as st
import os
import pickle
import numpy as np
from scipy.interpolate import interp1d
import os
import getpass
from scipy.interpolate import interpn
from src.unittests import check_shape
from scipy.ndimage import uniform_filter1d

def traj_smoothing(traj, window_size = 5):

    traj_f = traj.copy()
    if len(traj_f) < 3:
        window_size = 1
    traj_f[:,0] = uniform_filter1d(traj_f[:,0], size=window_size)
    traj_f[:,1] = uniform_filter1d(traj_f[:,1], size=window_size)
    traj_f[:,2] = uniform_filter1d(traj_f[:,2], size=window_size)
    return traj_f


def excecute_planning_step(engine, session_state, config):
    """
    Calculate trajectory planning for the given step

    :param engine: str [Evolutionary]
    :param session_state: dict
        streamlit session state
    """
    (
        session_state["OFV"],
        session_state["traj"],
        session_state["azi_l"],
        session_state["incl_l"],
        session_state["edge_limitation"],
    ) = engine.DE_step(
        config,
        OFV=session_state["OFV"],
        azi_l=session_state["azi_l"],
        incl_l=session_state["incl_l"],
        traj=session_state["traj"],
    )
    
    

    return session_state


def excecute_planning_step_init(engine, session_state, config):
    """
    Calculate trajectory planning for the initial step

    :param engine: str [Evolutionary]
    :param session_state: dict
        streamlit session state
    """
    (
        session_state["OFV"],
        session_state["traj"],
        session_state["azi_l"],
        session_state["incl_l"],
        session_state["edge_limitation"],
    ) = engine.DE_step(
        config,
        init_pos=[
            session_state["Z_RTFE"],
            session_state["Y_RTFE"],
            session_state["X_RTFE"],
        ],
        bounds=[
            (0, session_state["Az_constr"]),
            (0, session_state["Az_constr"]),
            (0, session_state["Az_constr"]),
            (0, session_state["Zen_constr"]),
            (0, session_state["Zen_constr"]),
            (0, session_state["Zen_constr"]),
        ],
    )
    
    return session_state


def excecute_planning_full(engine, session_state, config):
    """
    Calculate full trajectory for the provided parameters

    :param engine: str [Evolutionary]
    :param session_state: dict
        streamlit session state

    """
    length = config["length"]

    if session_state["traj"] == None:
        (
            session_state["OFV"],
            session_state["traj"],
            session_state["azi_l"],
            session_state["incl_l"],
            session_state["edge_limitation"],
        ) = engine.DE_step(
            config,
            OFV=session_state["OFV"],
            azi_l=session_state["azi_l"],
            incl_l=session_state["incl_l"],
            init_pos=[
                session_state["Z_RTFE"],
                session_state["Y_RTFE"],
                session_state["X_RTFE"],
            ],
            bounds=[
                (0, session_state["Az_constr"]),
                (0, session_state["Az_constr"]),
                (0, session_state["Az_constr"]),
                (0, session_state["Zen_constr"]),
                (0, session_state["Zen_constr"]),
                (0, session_state["Zen_constr"]),
            ],
        )

    while (
        (session_state["traj"][2][-1] <= engine.old_cube.shape[2] - length)
        and (session_state["traj"][1][-1] <= engine.old_cube.shape[1])
        and (session_state["traj"][0][-1] <= engine.old_cube.shape[0])
    ):
        (
            session_state["OFV"],
            traj,
            session_state["azi_l"],
            session_state["incl_l"],
            session_state["edge_limitation"],
        ) = engine.DE_step(
            config,
            OFV=session_state["OFV"],
            azi_l=session_state["azi_l"],
            incl_l=session_state["incl_l"],
            traj=session_state["traj"],
        )
        session_state["traj"] = traj
        if session_state["edge_limitation"]:
            st.error("You reached the edge of the productive reservoir")

        #session_state['traj'] = traj_smoothing(session_state['traj'])


def save_uploadedfile(uploadedfile, type="resistivity"):
    """
    :param uploadedfile:
        name of the uploading file
    :param type: str [resistivity, porosity, trajectory]
        choose the file you want to upload

    """

    if type == "resistivity":
        with open(os.path.join("data/raw/res/", uploadedfile.name), "wb") as f:
            f.write(uploadedfile.getbuffer())
    elif type == "porosity":
        with open(os.path.join("data/raw/por/", uploadedfile.name), "wb") as f:
            f.write(uploadedfile.getbuffer())
    elif type == "trajectory":
        with open(os.path.join("data/raw/well/", uploadedfile.name), "wb") as f:
            f.write(uploadedfile.getbuffer())
    elif type == "clay":
        with open(os.path.join("data/raw/clay/", uploadedfile.name), "wb") as f:
            f.write(uploadedfile.getbuffer())

    elif type == "permeability":
        with open(os.path.join("data/raw/perm/", uploadedfile.name), "wb") as f:
            f.write(uploadedfile.getbuffer())

    return st.success("Saved File:{} to server".format(uploadedfile.name), icon="✅")


def upload_selected_file(path):
    """
    :param path: str
      path to the selected file
    :return:
      uploaded pickle file

    """
    with open(os.path.join(path), "rb") as f:
        file = pickle.load(f)
    return file


def convert_param_to_small(param: str):
    if param == "Productivity potential":
        param_s = "prod_potential"
    elif param == "Oil saturation":
        param_s = "oil_saturation"
    elif param == "Resistivity":
        param_s = "resistivity"
    return param_s


def file_selector(folder_path, on_change):
    """

    :param folder_path:
    :param on_change:
    :return:
        path to the selected file
    """
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox("Select a file", filenames, on_change=on_change)
    if not selected_filename:
        st.error("No files in a folder")
    else:
        return os.path.join(folder_path, selected_filename)


def calculate_oil_saturation(session_state, type="Archie"):
    """

    :param session_state:  dict
        streamlit session state dictionary
    :return:
        oil saturation cube
    """
    Rw = session_state["Rw"]
    Rt = session_state["file"]
    a = session_state["a"]
    porosity = session_state["porosity_file"]
    m = session_state["m"]
    n = session_state["n"]
    error = check_shape(porosity, Rt)
    if type != "Archie":
        R_sh = session_state["R_sh"]
        V_sh = session_state["clay_cube"]
        Rc = 0.4 * R_sh
        F = session_state["F_factor"]
        error += check_shape(Rt, V_sh)
    
    if error > 0:
        st.error("Shapes of cubes don't match")

    if type == "Archie":
        Sw = (Rw / Rt * (a / (porosity / 100) ** m)) ** (1 / n)
        
    elif type == "Samandoux":
        shale_rel_1 = (V_sh / R_sh) ** 2
        shale_rel_2 = V_sh / R_sh
        sec_rel = (4 * porosity**m) / (a * Rw * Rt)
        Sw = (a * Rw) / (2 * porosity**m) * (np.sqrt(shale_rel_1 + sec_rel) - shale_rel_2)
    elif type == "Indonesia":
        # Цикл  
        Sw = np.zeros_like(V_sh)
        
                    
        Sw = np.sqrt(1 / Rt) / ((V_sh ** (1 - 0.5 * V_sh) / np.sqrt(R_sh)) + np.sqrt(porosity** m / (a * Rw)))
        Sw = np.nan_to_num(Sw * 100, 0)
    elif type == "Fertl":
        rel_1 = (a * Rw) / Rt
        rel_2 = ((a * V_sh) / 2) ** 2
        Sw = porosity ** (-m / 2) * (np.sqrt(rel_1 + rel_2) - (a * V_sh) / 2)

    elif type == "De Witte":
        y = V_sh * (1 / Rw + 1 / R_sh)
        right_hs = -y + np.sqrt(y**2 - (4 / Rw) * ((V_sh**2 / Rc)-1/Rt))
        left_hs = Rw / (2 * porosity)
        Sw = left_hs * right_hs
        Sw = np.nan_to_num(Sw, np.mean(Sw))

    elif type == "Hossin":
        Sw = np.sqrt(( 0.9/porosity) * (1/Rt + V_sh**2/Rc) * Rw)

    elif type == "Kamel":
        numerator = (V_sh * Rt) + np.sqrt(V_sh**2*Rt**2 + (4 * R_sh**2 * Rt)/(F * Rw * (1-V_sh)**2))
        denominator =  2 * R_sh * Rt / (F * Rw * (1 - V_sh)**2)
        Sw = numerator / denominator
    oil_sat = 1 - Sw
    return oil_sat


def HD_interp(session_state, point):
    """
    Planned trajectory interpolation function
    Params:
        :param session_state: dict

        :param point: float
            x coordinate

        :return:
        x coordinate, y interpolated coordinate, z interpolated coordinate

    """
    X = session_state["planned_traj"][:, 0]
    Y = session_state["planned_traj"][:, 1]
    Z = session_state["planned_traj"][:, 2]

    f = interp1d(X, Z)
    f2 = interp1d(X, Y)

    return point, float(f2(point)), float(f(point))


def КЕА(X, Y, Z, disp, MD):
    """
    Planned trajectory interpolation function
    Params:
        :param X: array
            x coordinates of planned well
        :param Y: array
            y coordinates of planned well
        :param Z:
            z coordinates of planned well
        :param disp: float
            measued depth

        :return:
    """
    # calculate the length of entire planned trajectory
    lengths = np.zeros_like(X)
    for i in range(1, len(X)):
        dX = X[i] - X[i - 1]
        dY = Y[i] - Y[i - 1]
        dZ = Z[i] - Z[i - 1]
        lengths[i] = lengths[i - 1] + np.sqrt(dX**2 + dY**2 + dZ**2)
    well_length = lengths[-1]

    # check that md < length of well
    if disp > X.max():
        st.error("Horizontal displacement is greater than well length")
    #   raise ValueError("MD is greater than well length")

    # check index of current step
    i = 1
    while lengths[i] < disp:
        i += 1

    # интерполируем координаты точки
    prev_length = lengths[i - 1]
    dX = X[i] - X[i - 1]
    dY = Y[i] - Y[i - 1]
    dZ = Z[i] - Z[i - 1]
    x1 = X[i - 1] + (X[i] - X[i - 1]) * (MD - prev_length) / (lengths[i] - prev_length)
    y1 = Y[i - 1] + (Y[i] - Y[i - 1]) * (MD - prev_length) / (lengths[i] - prev_length)
    z1 = Z[i - 1] + (Z[i] - Z[i - 1]) * (MD - prev_length) / (lengths[i] - prev_length)

    return x1, y1, z1, well_length


def create_cache_folder():
    # Получаем имя пользователя, от которого запускается код
    user_name = getpass.getuser()

    # Формируем путь к папке .streamlit
    streamlit_path = os.path.join("C:", os.sep, "Users", user_name, ".streamlit")

    # Создаем папку .streamlit, если она не существует
    if not os.path.exists(streamlit_path):
        os.mkdir(streamlit_path)

    # Формируем путь к папке cache внутри .streamlit
    cache_path = os.path.join(streamlit_path, "cache")

    # Создаем папку cache или перезаписываем ее, если она уже существует
    if os.path.exists(cache_path):
        os.system(f'rd /s /q "{cache_path}"')  # Удаляем папку и ее содержимое
    os.mkdir(cache_path)


def calculate_OFV_planned(data, session_state, x_coord):
    """
    Params:
    data: 3d array
      cube of parameters
    session_state: dict
      streamlit session state
    x_coord: float
      x coordinate


    """
    x1, y1, z1 = HD_interp(session_state, x_coord)
    # calculate OFV for planned_traj
    x = np.linspace(0, data.shape[0] - 1, data.shape[0])
    y = np.linspace(0, data.shape[1] - 1, data.shape[1])
    z = np.linspace(0, data.shape[2] - 1, data.shape[2])
    points = (x, y, z)
    OFV = (
        interpn(
            points,
            session_state["oil_saturation_file"],
            [x1 - session_state['x_cube'], y1 - session_state["y_cube"], z1 - session_state['z_cube']],
            method="nearest",
        )
        * 10
    )
    return OFV



def calc_OFV(traj_corr, session_state, cube):
    traj_renewed = traj_corr.copy()
    traj_renewed[0] = traj_corr[2]
    traj_renewed[2] = traj_corr[0]
    x = np.linspace(0, cube.shape[0] - 1, cube.shape[0])
    y = np.linspace(0, cube.shape[1] - 1, cube.shape[1])
    z = np.linspace(0, cube.shape[2] - 1, cube.shape[2])
    points = (x, y, z)
    OFV_corr = 0

    for i in range(len(traj_corr[0])):
      #  x1, y1, z1 = HD_interp(session_state, traj_renewed[0][-i])
        z_cor, y_cor, x_cor = traj_corr[0][-i], traj_corr[1][-i], traj_corr[2][-i]
        try:
            OFV_corr += interpn(points, cube, [x_cor, y_cor, z_cor], method="nearest")
            c += 1
        except:
                continue
    return OFV_corr
import streamlit as st
import numpy as np
import random

def check_well_types(file):
    """
    Check type of well
    """
    if isinstance(file, np.ndarray):
        return file
    else:
        st.error("Trajectory must be an numpy array. The uploading format is not available")


def check_planned_well(session_state):
    """
    Error assertion function for one file
    """
    error = False
    try:
        assert "planned_traj" in session_state.keys()
    except:
        st.error("Planned trajectory is not loaded")
        error = True
    return error


def check_types(file, session_state):
    """
    Params:
        :param file: str or dict
            uploaded cube of parameters
        :param session_state: dict
            streamlit session state
    :return:
        numpy array of parameters, cube initial coordinates
    """
    if isinstance(file, np.ndarray):
        st.subheader("Assign start coordinates for the cube in meters ")
        with st.form(key=f"my_form_{random.random()}"):
            session_state["x_cube"] = st.number_input("X, m", min_value=0.0, max_value=1000000.0, value=0.0, step=0.1)
            session_state["y_cube"] = st.number_input("Y, m", min_value=0.0, max_value=1000000.0, value=0.0, step=0.1)
            session_state["z_cube"] = st.number_input("Z, m", min_value=0.0, max_value=1000000.0, value=0.0, step=0.1)

            submit_button = st.form_submit_button(label="Submit")
        return (
            file,
            session_state["x_cube"],
            session_state["y_cube"],
            session_state["z_cube"],
        )
    elif isinstance(file, dict):
        file_m = file["data"]
        session_state["x_cube"] = file["x_coords"][0]
        session_state["y_cube"] = file["y_coords"][0]
        session_state["z_cube"] = file["z_coords"][0]
        return (
            file_m,
            session_state["x_cube"],
            session_state["y_cube"],
            session_state["z_cube"],
        )


def check_file(session_state, include="oil_saturation"):
    """
    Error assertion function for one file
    """
    error = False
    if include == "resistivity":
        try:
            assert "file_path" in session_state.keys()
            assert "file" in session_state.keys()
        except:
            st.error("Resistivity data is not loaded. Please, " "upload resistivity cube and try again")
            error = True
    if include == "porosity":
        try:
            assert "porosity_file_path" in session_state.keys()
            assert "porosity_file" in session_state.keys()
        except:
            st.error("Porosity is not loaded. Please, " "upload porosity cube and try again")
            error = True
    if include == "oil_saturation":
        try:
            assert "oil_saturation_file" in session_state.keys()
        except:
            st.error("Oil saturation was not calculated. Please, " "calculate oil saturation cube")
            error = True
    if include == "permeability":
        try:
            assert "permeability_file" in session_state.keys()
        except:
            st.error("Permeability file was not found. Please, upload permeability cube ")

            error = True
    if include == "shale":
        try:
            assert "clay_cube" in session_state.keys()
        except:
            st.error("Clay content data is not loaded. Please, " "upload Clay content cube and try again")
            error = True

    if include == "productivity_potential":
        try:
            assert "productivity_potential" in session_state.keys()
        except:
            st.error("Productivity potential file was not found. Please, upload productivity potential cube ")

            error = True

    return error


def file_path_assert(session_state, exclude=["oil_saturation"]):
    """
    Error assertion function
    """
    error = False
    if "resistivity" not in exclude:
        try:
            assert "file_path" in session_state.keys()
            assert "file" in session_state.keys()
        except:
            st.error("Resistivity data is not loaded. Please, " "upload resistivity cube and try again")
            error = True

    if "shale" not in exclude:
        try:
            assert "clay_cube" in session_state.keys()
        except:
            st.error("Clay content data is not loaded. Please, " "upload Clay content cube and try again")
            error = True
    if "porosity" not in exclude:
        try:
            assert "porosity_file_path" in session_state.keys()
            assert "porosity_file" in session_state.keys()
        except:
            st.error("Porosity is not loaded. Please, " "upload porosity cube and try again")
            error = True
    if "oil_saturation" not in exclude:
        try:
            assert "oil_saturation_file" in session_state.keys()
        except:
            st.error("Oil saturation was not calculated. Please, " "calculate oil saturation cube")
            error = True
    if "permeability" not in exclude:
        try:
            assert "permeability_file" in session_state.keys()
        except:
            st.error("Permeability file was not found. Please, upload permeability cube ")

            error = True
    if "prod_potential" not in exclude:
        try:
            assert "prod_potential" in session_state.keys()
        except:
            st.error("Productivity potential file was not found. Please, upload productivity potential cube ")

            error = True

    return error


def params_assert(key, session_state):
    """
    Error assertion function
    """
    error = False
    try:
        assert key in session_state.keys()
    except:
        st.error(
            f'You didn"t assigned parameters in "Set parameters" section. Click ' f"on set button on subheader panel"
        )
        error = True
    return error


def check_loads(session_state):
    try:
        perm = session_state.permeability_file
    except:
        st.warning("No permeability loaded")
    try:
        res = session_state.file
    except:
        st.error("No resistivity loaded")
    try:
        por = session_state.porosity_file
    except:
        st.error("No porosity loaded")
    try:
        clay = session_state.clay_cube
    except:
        st.error("No clay loaded")

def check_shape(arr1, arr2):
    error = False
    if arr1.shape != arr2.shape:
        st.error("Shapes do not match. Check the shapes of two cubes")
        error = True
    return error
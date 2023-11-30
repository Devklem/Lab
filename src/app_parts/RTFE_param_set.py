import sys

import streamlit as st

sys.path.append("./streamlit_app")
from src.states.states import callback_set
from src.utils import HD_interp, convert_param_to_small
from src.unittests import file_path_assert, check_file, check_planned_well


def RTFE_parameters_set_section(session_state):
    st.sidebar.subheader("RTFE initial parameters")

    st.sidebar.button("Set", on_click=callback_set)
    if session_state.button_set_clicked:
        if "parameter_for_opt"  in session_state:
            
            error = check_file(
                session_state,
                include=[
                    "resisitivty",
                    "porosity",
                    convert_param_to_small(session_state["parameter_for_opt"]),
                ],
            )
            error2 = check_planned_well(session_state)

            if not error and not error2:
                st.subheader("Setting initialization parameters")
                with st.form(key="my_form2"):
                    session_state["engine"] = "Evolutionary algorithm"

                    Rt = session_state["oil_saturation_file"].copy().T

                    shape_x = session_state["x_cube"] + Rt.shape[0]
                    shape_y = session_state["y_cube"] + Rt.shape[1]
                    shape_z = session_state["z_cube"] + Rt.shape[2]

                    session_state["HD"] = st.number_input(
                        "Horizontal displacement [m]",
                        min_value=session_state["x_cube"],
                    
                        max_value=float(session_state["planned_traj"][:, 0].max()),
                        value=session_state["x_cube"] + 20.0,
                        step=0.01,
                    )

                    X, Y, Z = (
                        session_state["planned_traj"][:, 0],
                        session_state["planned_traj"][:, 1],
                        session_state["planned_traj"][:, 2],
                    )

                    x1, y1, z1 = HD_interp(session_state, session_state["HD"])
                    session_state["X_RTFE"] = x1
                    session_state["Y_RTFE"] = y1
                    session_state["Z_RTFE"] = z1

                    session_state["X_RTFE"] -= session_state["x_cube"]
                    session_state["Y_RTFE"] -= session_state["y_cube"]
                    session_state["Z_RTFE"] -= session_state["z_cube"]

                    session_state["Az_i_RTFE"] = st.number_input(
                        "Initial azimuth [°]",
                        min_value=-180.0,
                        max_value=180.0,
                        value=0.0,
                        step=0.1,
                    )
                    session_state["Zen_i_RTFE"] = st.number_input(
                        "Initial zenith [°]",
                        min_value=0.0,
                        max_value=180.0,
                        value=0.0,
                        step=0.1,
                    )
                    session_state["Az_constr"] = st.number_input(
                        "Azimuth constraint [°]",
                        min_value=0.0,
                        max_value=180.0,
                        value=180.0,
                        step=0.01,
                    )
                    session_state["Zen_constr"] = st.number_input(
                        "Zenith constraint [°]",
                        min_value=0.0,
                        max_value=92.0,
                        value=92.0,
                        step=0.01,
                    )
                    session_state["DL_constr"] = st.number_input(
                        "Dogleg constraint [°/10 m]",
                        min_value=-180.0,
                        max_value=180.0,
                        value=3.0,
                        step=0.1,
                    )
                    session_state["Step_L"] = st.number_input(
                        "Length of 1 step [m]",
                        min_value=0.0,
                        max_value=20.0,
                        value=5.0,
                        step=1.0,
                    )
                    session_state["c"] = 0
                    session_state["OFV"] = 0

                    st.form_submit_button(label="Submit")

                if session_state["engine"] == "Evolutionary algorithm":
                    session_state["DE_maxiter"] = 1200
                    session_state["DE_popsize"] = 400
                    session_state["DE_mutation"] = 0.2
                    st.subheader("Choose parameters of differential evolution algorithm (Optional)")

                    engine = st.selectbox(
                        "Select optimizer",
                        (
                            "Differential evolution",
                            "Particle swarm optimization",
                            "Chaotic particle swarm optimization",
                        ),
                    )
                    engine_dict = {
                        "Differential evolution": "de",
                        "Particle swarm optimization": "pso",
                        "Chaotic particle swarm optimization": "cpso",
                    }
                    session_state["engine"] = engine_dict[engine]
                    with st.form(key="DE_params"):
                        if session_state["engine"] == "de":
                            session_state["DE_mutation"] = st.number_input(
                                "Mutation constant",
                                min_value=0.0,
                                max_value=1.0,
                                value=0.7,
                                step=0.01,
                            )

                            session_state["DE_strategy"] = st.selectbox(
                                "Differential evolution strategy",
                                (
                                    "best1bin",
                                    "best1exp",
                                    "rand1exp",
                                    "randtobest1exp",
                                    "currenttobest1exp",
                                    "best2exp",
                                    "rand2exp",
                                    "randtobest1bin",
                                    "currenttobest1bin",
                                    "best2bin",
                                    "rand2bin",
                                    "rand1bin",
                                ),
                            )
                        session_state["DE_maxiter"] = st.number_input(
                            "Maximum number of iterations",
                            min_value=100,
                            max_value=100000,
                            value=1000,
                            step=1,
                        )

                        session_state["DE_popsize"] = st.number_input(
                            "Population size",
                            min_value=10,
                            max_value=10000000,
                            value=100,
                            step=1,
                        )

                        st.form_submit_button(label="Submit")

        else:
            st.error("You did not calculate target parameter")

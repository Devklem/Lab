import sys

import pandas as pd
import streamlit as st

sys.path.append("./Geosteering")
from src.planning import DE_algo, plot_results_2, RTFE_main, config
from src.utils import (
    excecute_planning_step,
    excecute_planning_full,
    excecute_planning_step_init,
    calc_OFV
)
from src.states.states import callback_launch, callback_step_planning
from src.unittests import file_path_assert, params_assert, check_file
from src.visualization import plot_well_cube, plot_well_cube_sec

parameter_to_load = {
    "Productivity potential": "prod_potential",
    "Oil saturation": "oil_saturation",
}

def simulation_section(session_state):
    st.sidebar.subheader("RTFE simulation")
    st.sidebar.button("Simulation", on_click=callback_launch)

    if session_state.button_launch_clicked:
        st.subheader("RTFE simulation")

        error2 = params_assert("X_RTFE", session_state)
        error3 = params_assert("DE_popsize", session_state)
        col1, col2 = st.columns(2)
        with col1:

            parameter_for_opt = st.radio(
                "Choose parameter for optimization",
                ["Oil saturation", "Productivity potential"],
            )
        with col2:
            st.markdown("Selected parameters")
            data = [
                round(session_state.X_RTFE + session_state["x_cube"], 2),
                round(session_state.Y_RTFE + session_state["y_cube"], 2),
                round(session_state.Z_RTFE + session_state["z_cube"], 2),
                session_state["Az_i_RTFE"],
                int(session_state["Zen_i_RTFE"]),
                int(session_state["Az_i_RTFE"]),
                session_state["DL_constr"],
                session_state["Step_L"],
            ]
            df = pd.DataFrame(data)
            df.index = [
                "X coordinate [m]",
                "Y coordinate [m]",
                "Z coordinate [m]",
                "Initial azimuth [deg]",
                "Initial zenith [deg]",
                "Azimuth constraint [deg]",
                "Dogleg constraint [deg/10 m]",
                "Step legnth [m]",
            ]
            st.dataframe(df)

       

        error = check_file(session_state, include=parameter_to_load[parameter_for_opt])
       
        if not error2 and not error3 and not error:
            if parameter_for_opt == "Oil saturation":
                cube = session_state.oil_saturation_file.copy()
                cube_to_plot = "oil_saturation"
            elif parameter_for_opt == "Productivity potential":
                cube = session_state.prod_potential.copy()
                cube_to_plot = "productivity_potential"
            elif parameter_for_opt == "Resistivity":
                cube = session_state.file.copy()
                cube_to_plot = "resistivity"

            engine = RTFE_main(cube)
            

            # config for planner
            config = {
                    "solver": session_state['engine'],
                    "strategy": session_state['DE_strategy'],
                    "pop_size": session_state['DE_popsize'],
                    "maxiter": session_state['DE_maxiter'],
                    "F": 0.7,
                    "cr": 0.7,
                    "angle_constraint": session_state['DL_constr'],
                    "length": session_state['Step_L'],
                }

            
            col1, col2, col3 = st.columns(3)
            with col2:
                st.session_state.full_planning_clicked = st.button(
                    "Start full planning"
                )  # on_click=callback_full_planning)
            with col1:
                session_state.step_planning_clicked = st.button(
                    "Step planning",
                    disabled=session_state.disabled,
                    on_click=callback_step_planning,
                )
            with col3:
                reset = st.button("Reset trajectory")
            if reset:
                session_state["traj"] = None
                session_state["OFV"] = 0
                session_state["c"] = 0
                session_state["azi_l"] = None
                session_state["incl_l"] = None
                session_state["traj_ready"] = False
                session_state.disabled = False

            if st.session_state.full_planning_clicked:
                session_state["traj"] = None
             
                engine = RTFE_main(cube)
                with st.spinner("Planning the trajectory.."):
                    excecute_planning_full(engine, session_state, config)
                    corr_ofv = calc_OFV(session_state['traj'], st.session_state, cube)

                    #st.success(f'Objective function value : {round(float(corr_ofv), 2)}')
                    session_state["ready_engine"] = engine
                    session_state["OFV"] = round(float(session_state["OFV"]), 2)
                        
                    # plot_well_cube(
                    #     session_state,
                    #     plot_plan="Yes",
                    #     dot=True,
                    #     corrected_traj=True,
                    #     cube_to_plot=cube_to_plot,
                    #     )
                    plot_well_cube_sec(
                        session_state,
                        plan_well=None,
                        plot_plan=True,
                        dot=True,
                        plot_corrected=True,
                        corrected_traj=session_state["traj"],
                        opacity=0.5,
                        cube_to_plot=cube_to_plot,
                        )

            if session_state.step_planning_clicked:
                try:
                    cond = session_state["traj"][2][-1] >= cube.shape[2] - 1.1 * session_state["Step_L"]
                except:
                    cond = False

                if cond or session_state["traj_ready"]:
                    session_state.disabled = True
                    st.write("Trajectory has already built")

                    
                  #  st.success(f'Objective function value : {round(float(session_state["OFV"]), 2)}')

                    # plot_well_cube(
                    #     session_state,
                    #     plot_plan="Yes",
                    #     dot=True,
                    #     corrected_traj=True,
                    #     cube_to_plot=cube_to_plot,
                    # )

                    plot_well_cube_sec(
                        session_state,
                        plan_well=None,
                        plot_plan=True,
                        dot=True,
                        plot_corrected=True,
                        corrected_traj=session_state["traj"],
                        opacity=0.5,
                        cube_to_plot=cube_to_plot,
                    )
                else:
                    with st.spinner("Planning the trajectory.."):
                        if session_state["c"] == 0:
                            session_state = excecute_planning_step_init(engine, session_state, config)
                        else:
                            session_state = excecute_planning_step(engine, session_state, config)
                        session_state["c"] += 1
                        corr_ofv = calc_OFV(session_state['traj'], st.session_state, cube)
                        #   st.success(f'Objective function value : {round(float(corr_ofv), 2)}')

                        if session_state["traj"][2][-1] >= cube.shape[2] - 1.1 * session_state["Step_L"]:
                            session_state.disabled = True

                        # plot_well_cube(
                        #     session_state,
                        #     plot_plan="Yes",
                        #     dot=False,
                        #     corrected_traj=True,
                        #     cube_to_plot=cube_to_plot,
                        # )

                        plot_well_cube_sec(
                            session_state,
                            plan_well=None,
                            plot_plan=True,
                            dot=True,
                            plot_corrected=True,
                            corrected_traj=session_state["traj"],
                            opacity=0.5,
                            cube_to_plot=cube_to_plot,
                        )

            # if  'traj' in session_state.keys():
            #     fig, fig3, fig2_1, fig2_2 = plot_results_2(cube, session_state['traj'])
            #     st.plotly_chart(fig)
            #     st.plotly_chart(fig3)
            #     fig, fig3, fig2_1, fig2_2 = plot_results_2(session_state['file'], session_state['traj'], data='resistivity')
            #     st.plotly_chart(fig)
            #     st.plotly_chart(fig3)

    else:
        st.sidebar.write("")

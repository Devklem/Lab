import sys

import streamlit as st

sys.path.append("./streamlit_app")
sys.path.append("./Geosteering")
from src.states import callback
from src.unittests import file_path_assert, check_file
from src.visualization.main_vis import plot_vis
from src.visualization.visualize import plot_well_cube


def data_vis_section(session_state):
    st.sidebar.subheader("Data visualization")
    

    session_state["plot_button"] = st.sidebar.button("Plot LWD data", disabled=False, on_click=callback)
    if session_state.button_clicked:
        session_state["vis_type"] = st.radio(
            "Choose type of plotting:", ["2D", "3D"], on_change=callback, key="oil_sat"
        )
        op_slider = 1
        plot_plan = st.radio("Plot planned well", ["No", "Yes"], on_change=callback)
        colormap = st.selectbox("Choose a colormap", ["rainbow", "viridis"])

        if session_state["vis_type"] == "3D":
            op_slider = st.slider("Choose the opacity", min_value=0.1, max_value=1.0, on_change=callback)

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            [
                "Resitivity",
                "Porosity",
                "Clay content",
                "Oil saturation",
                "Permeability",
                "Productivity potential",
            ]
        )
        with tab1:
                error = file_path_assert(
                    session_state,
                    exclude=[
                        "parameter_for_opt",
                        "shale",
                        "prod_potential",
                        "permeability",
                        "oil_saturation",
                    ],
                )
                if not error:
                    plot_vis(
                        session_state,
                        callback,
                        plot_plan=plot_plan,
                        op_slider=op_slider,
                        plot_param="resistivity",
                        colormap=colormap,
                    )

        with tab2:
                error = file_path_assert(
                    session_state,
                    exclude=[
                        "parameter_for_opt",
                        "shale",
                        "prod_potential",
                        "permeability",
                        "oil_saturation",
                    ],
                )
                if not error:
                    plot_vis(
                        session_state,
                        callback,
                        plot_plan=plot_plan,
                        op_slider=op_slider,
                        plot_param="porosity",
                        colormap=colormap,
                    )
           
        with tab3:
                error = file_path_assert(
                    session_state,
                    exclude=[
                        "parameter_for_opt",
                        "prod_potential",
                        "permeability",
                        "oil_saturation",
                    ],
                )
                if not error:
                    plot_vis(
                        session_state,
                        callback,
                        plot_plan=plot_plan,
                        op_slider=op_slider,
                        plot_param="shale",
                        colormap=colormap,
                    )

        with tab4:
            error = file_path_assert(
                session_state,
                exclude=[
                    "parameter_for_opt",
                    "prod_potential",
                    "permeability",
                ],
            )
            if not error:
                plot_vis(
                    session_state,
                    callback,
                    plot_plan=plot_plan,
                    op_slider=op_slider,
                    plot_param="oil_saturation",
                    colormap=colormap,
                )
        with tab5:
            error = file_path_assert(
                session_state,
                exclude=[
                    "parameter_for_opt",
                    "prod_potential",
                ],
            )

            if not error:
                plot_vis(
                    session_state,
                    callback,
                    plot_plan=plot_plan,
                    op_slider=op_slider,
                    plot_param="permeability",
                    colormap=colormap,
                )

        with tab6:
                error = file_path_assert(session_state, exclude=["parameter_for_opt"])

                if not error:
                    plot_vis(
                        session_state,
                        callback,
                        plot_plan=plot_plan,
                        op_slider=op_slider,
                        plot_param="productivity_potential",
                        colormap=colormap,
                    )

    else:
        st.sidebar.write("")

import streamlit as st

from .visualize import vis_2d, plotly_3d_cube, plot_well_cube


def plot_vis(
    session_state,
    callback,
    plot_plan=True,
    op_slider=0.3,
    plot_param="porosity",
    colormap="rainbow",
):
    if plot_param == "porosity":
        data = session_state.porosity_file
    elif plot_param == "resistivity":
        data = session_state.file
    elif plot_param == "oil_saturation":
        data = session_state.oil_saturation_file * 100
    elif plot_param == "shale":
        data = session_state.clay_cube * 100
    elif plot_param == "permeability":
        data = session_state.permeability_file
    elif plot_param == "productivity_potential":
        data = session_state.prod_potential

    if session_state["vis_type"] == "2D":
        session_state.projection = st.selectbox(
            "Choose a plane for plotting:",
            ["Slice", "Inline", "Xline"],
            on_change=callback,
            key=plot_param,
        )
        column1, column2 = st.columns([1, 4])
        vis_2d.clear()
        if session_state.projection == "Slice":
            # Add slider to column 1
            slider = column1.slider(
                "Slice #",
                min_value=1,
                max_value=data.shape[0],
                on_change=callback,
                value=1,
                key=plot_param + "slider",
            )
            # Add plot to column 2
            slider -= 1
            vis_2d(data, slider, plot_param, axis="Slice", colormap=colormap)
            vis_2d.clear()

        elif session_state.projection == "Inline":
            # Add slider to column 1
            slider = column1.slider(
                "Inline",
                min_value=1,
                on_change=callback,
                max_value=data.shape[1],
                value=1,
                key=plot_param + "slider2",
            )
            slider -= 1
            vis_2d(data, slider, plot_param, axis="Inline", colormap=colormap)
            vis_2d.clear()

        elif session_state.projection == "Xline":
            # Add slider to column 1

            slider = column1.slider(
                "Xline",
                min_value=1,
                on_change=callback,
                max_value=data.shape[2],
                value=1,
                key=plot_param + "slider3",
            )
            # Add plot to column 2
            slider -= 1
            vis_2d(data, slider, plot_param, axis="Xline", colormap=colormap)
            vis_2d.clear()
        #################################################################################

    elif session_state["vis_type"] == "3D":
        plotly_3d_cube.clear()


        if plot_plan == "Yes" and "planned_traj" not in session_state:
            st.error('No planned trajectory found')
        else:
            button_accept = st.button("Plot 3D cube", key=plot_param)

            if button_accept:
                plot_well_cube(
                    session_state,  
                    plot_plan, 
                    opacity=op_slider,
                    cube_to_plot=plot_param,
                    plot_vis=False, 
                )


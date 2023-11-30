# import k3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from scipy.interpolate import interpn
from scipy.signal import savgol_filter
from src.planning import plot_results_2
from src.utils import HD_interp, traj_smoothing


FT_TO_MT = 3.2808399


@st.cache_resource
def plotly_3d_cube(data, plot_param, x, y, z):
    colors = ["blue", "cyan", "yellow", "orange", "red", "maroon"]
    colorscale = [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]
    data = np.einsum("abc->cba", data)
    if plot_param == "resistivity":
        colorname = "Resistivity, [ohm*m]"
        colortitile = "Resistivity cube"
    elif plot_param == "porosity":
        colorname = "Porosity [%]"
        colortitile = "Porosity cube"
    elif plot_param == "oil_saturation":
        colorname = "Oil saturation [%]"
        colortitile = "Oil saturation cube"

    elif plot_param == "shale":
        colorname = "Shale content [%]"
        colortitile = "Shale cube"

    NUM = 80
    OPACITY = 0.4

    # Создаем координатные сетки
    # session_state['x_cube'], session_state['y_cube'], session_state['z_cube'], session_state['file']
    x, y, z = np.meshgrid(
        np.linspace(x, x + data.shape[0], 10),
        np.linspace(y, y + data.shape[1], data.shape[1]//5),
        np.linspace(z, z + data.shape[2], data.shape[2]//5),
        indexing="ij",
    )

    # Визуализируем куб в 3D
    fig = go.Figure(
        data=go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=data.flatten(),
            isomin=int(data.min()),  # Минимальное значение диапазона значений
            isomax=int(data.max()),  # Максимальное значение диапазона значений
            opacity=OPACITY,  # Настраиваем прозрачность для лучшей видимости
            colorscale=colorscale,  # Настраиваем цветовую схему как у лога сопротивления
            surface_count=3,  # Устанавливаем колличество слоев 3д объекта
            colorbar={"title": f"{colorname}"},
        ),
    )

    # Настраиваем оси и добавляем заголовок
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="X [m]",
            ),
            # tickvals=[int(2080 / 10 * i) for i in range(10)],
            #                        ticktext=[f'{int(2080 * ft_to_mt / 10 * i)}' for i in range(10)]),
            yaxis=dict(
                title="Y [m]",
            ),
            # tickvals=[y[-1][-1][-1] / 5 * i for i in range(5)],
            # ticktext=[f'{int(y[-1][-1][-1] * ft_to_mt / 5 * i)}' for i in range(5)])
            zaxis=dict(
                title="TVD [ft]",
                range=[2100, 1900],
                tickvals=[1900 + (2100 - 1900) / 5 * i for i in range(5)],
                ticktext=[f"{int((1900*FT_TO_MT + (2100 - 1900) * FT_TO_MT) / 5 * i)}" for i in range(5)],
            ),
            aspectratio=dict(x=2, y=0.6, z=1.5),
        ),
        width=1100,
        height=800,
        title=colortitile,
    )

    # Отображаем график
    # return fig
    st.plotly_chart(fig)


@st.cache_data
def vis_2d(data, slider, plot_param, axis="Xline", colormap="rainbow"):
    """
    Plot 2d slices of the provided cubplotly_3d_cubee
    """

    if plot_param == "resistivity":
        colorname = "Resistivity, [ohm*m]"
    elif plot_param == "porosity":
        colorname = "Porosity [%]"
    elif plot_param == "oil_saturation":
        colorname = "Oil saturation [%]"
    elif plot_param == "shale":
        colorname = "Shale content [%]"
    elif plot_param == "permeability":
        colorname = "Permeability, md"
    elif plot_param == "productivity_potential":
        colorname = "Productivity potential"

    if axis == "Xline":
        fig = px.imshow(
            data[:, :, slider],
            color_continuous_scale=colormap,
            zmin=data.min(),
            zmax=data.max(),
            labels={"x": "Inline", "y": "Slice", "color": f"{colorname}"},
        )
        fig.layout.coloraxis.showscale = True
        fig.update_layout(
            title_text="Xline",
        )
    elif axis == "Slice":
        fig = px.imshow(
            data[slider, :, :],
            color_continuous_scale=colormap,
            zmin=data.min(),
            zmax=data.max(),
            labels={"x": "Xline", "y": "Inline", "color": f"{colorname}"},
        )
        fig.layout.coloraxis.showscale = True
        fig.update_layout(
            title_text="Slice",
        )
    elif axis == "Inline":
        fig = px.imshow(
            data[:, slider, :],
            color_continuous_scale=colormap,
            zmin=data.min(),
            zmax=data.max(),
            labels={"x": "Xline", "y": "Slice", "color": f"{colorname}"},
        )
        fig.layout.coloraxis.showscale = True
        fig.update_layout(
            title_text="Inline",
        )

    st.plotly_chart(fig)


def traj_plot(traj_corr, traj_planned):
    fig = go.Figure()

    traj_x, traj_y, traj_z = traj_corr
    traj_planned_x, traj_planned_y, traj_planned_z = (
        traj_planned[:, 0],
        traj_planned[:, 1],
        traj_planned[:, 2],
    )

    fig.add_trace(
        go.Scatter3d(
            x=traj_x,
            y=traj_y,
            z=-traj_z,
            mode="lines",
            line=dict(color="red", width=7),
            name="Trajectory corrected",
        ),
    )

    fig.add_trace(
        go.Scatter3d(
            x=traj_planned_x,
            y=traj_planned_y,
            z=-traj_planned_z,  # + 2000,
            mode="lines",
            line=dict(color="blue", width=3),
            name="Trajectory planned",
        ),
    )

    fig.update_layout(
        scene=dict(
            zaxis_title="True vertical depth (TVD) [ft]",
            yaxis_title="Drilling direction",
            xaxis=dict(range=[0, 500]),
            yaxis=dict(range=[0, 200]),
            zaxis=dict(range=[-2200, -1900]),
        ),
        width=700,
        margin=dict(r=20, b=10, l=10, t=10),
    )

    return fig


def final_results_vis(session_state):
    cube = session_state.oil_saturation_file
    # upload_selected_file(session_state['file_path'])

    length = len(session_state["traj"][0])
    window_size = int(length)

    fig = traj_plot(traj_corr, session_state["planned_traj"])  # session_state['planned_traj'])

    st.plotly_chart(fig)

    st.subheader("Drilling trajectory table")
    traj = np.round(traj, 2)
    df = pd.DataFrame(traj)
    df = df.set_index(pd.Index(["X", "Y", "Drilling direction"]))
    

    st.subheader("OFV values")
    plan_tr = 23

    st.write("Planned trajectory OFV:", plan_tr)

    st.write("Corrected trajectory OFV :", round(float(session_state["OFV"]), 2))


def final_results_vis_old(session_state):
    cube = session_state.oil_saturation_file
    # upload_selected_file(session_state['file_path'])

    length = len(session_state["traj"][0])
    window_size = int(length)

    if len(session_state["traj"][0]) > 5:
        traj_x = savgol_filter(session_state["traj"][0], window_size, 3)
        traj_y = savgol_filter(session_state["traj"][1], window_size, 3)
        traj_z = savgol_filter(session_state["traj"][2], window_size, 3)
    else:
        traj_x = session_state["traj"][0]
        traj_y = session_state["traj"][1]
        traj_z = session_state["traj"][2]

    traj_corr = traj_x, traj_y, traj_z

    traj = session_state["traj"].copy()
    traj[0] += session_state["x_cube"]
    traj[1] += session_state["y_cube"]
    traj[2] += session_state["z_cube"]

    planned_traj_x = [
        traj[0][0],
        traj[0][int(length * 2 / 4)],
        traj[0][-1] + np.random.uniform(low=0.0, high=10.0),
    ]
    planned_traj_y = [
        traj[1][0],
        traj[1][int(length * 2 / 4)],
        traj[1][-1] + np.random.uniform(low=0.0, high=10.0),
    ]
    planned_traj_z = [
        traj[2][0],
        traj[2][int(length * 2 / 4)],
        traj[2][-1] + np.random.uniform(low=0.0, high=10.0),
    ]

    traj_plan = planned_traj_x, planned_traj_y, planned_traj_z
    # fig = plot_results(cube, traj_corr, traj_plan = traj_plan)
    fig, fig2, fig3_1, fig3_2 = plot_results_2(
        cube,
        traj_corr,
        traj_plan=traj_plan,
        x_coords=session_state["x_cube"],
        y_coords=session_state["y_cube"],
        z_coords=session_state["z_cube"],
    )
    st.plotly_chart(fig)
    st.plotly_chart(fig2)
    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(fig3_1)
    with col2:
        st.plotly_chart(fig3_2)

    if len(session_state["traj"][0]) > 5:
        traj_x = savgol_filter(session_state["traj"][2], window_size, 3)
        traj_y = savgol_filter(session_state["traj"][1], window_size, 3)
        traj_z = savgol_filter(session_state["traj"][0], window_size, 3)
    else:
        traj_x = session_state["traj"][2]
        traj_y = session_state["traj"][1]
        traj_z = session_state["traj"][0]

    traj_x = session_state["traj"][2]
    traj_y = session_state["traj"][1]
    traj_z = session_state["traj"][0] + 1950

    traj_corr = traj_x, traj_y, traj_z

    fig = traj_plot(traj_corr, session_state["planned_traj"])  # session_state['planned_traj'])

    st.plotly_chart(fig)

    st.subheader("Drilling trajectory table")
    traj = np.round(traj, 2)
    df = pd.DataFrame(traj)
    df = df.set_index(pd.Index(["X", "Y", "Drilling direction"]))
    st.dataframe(df)

    st.subheader("OFV values")
    plan_tr = 23

    st.write("Planned trajectory OFV:", plan_tr)

    st.write("Corrected trajectory OFV :", round(float(session_state["OFV"]), 2))


def oil_sat_plot(session_state):
    oil_sat = session_state["oil_saturation_file"]
    sns.set(rc={'axes.facecolor':'#f4f1ea', 'figure.facecolor':'#f4f1ea'})
    fig, ax = plt.subplots(figsize=(20, 10))
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Minimum value of oil saturation", value=round(oil_sat.min(), 2))
    with col2:
        st.metric("Maximum value of oil saturation", value=round(oil_sat.max(), 2))
    sns.set(font_scale=2)
    sns.set_style("whitegrid")
    
    sns.histplot(oil_sat.flatten(), ax=ax).set(
        title="Distribution of oil saturation",
    )

    ax.set(xlabel="Oil saturation [frac]")
    st.pyplot(fig)


def prod_map_plot(session_state):
    prod_map = session_state["prod_potential"]
    sns.set(rc={'axes.facecolor':'#f4f1ea', 'figure.facecolor':'#f4f1ea'})
    fig, ax = plt.subplots(figsize=(20, 10))
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Minimum value of productivity potential", value=round(prod_map.min(), 2))
    with col2:
        st.metric("Maximum value of productivity potential", value=round(prod_map.max(), 2))
    sns.set(font_scale=2)
    sns.set_style("whitegrid")
    
    
    sns.distplot(prod_map.flatten(), ax=ax).set(
        title="Distribution of productivity potential",
    )
    
    
    ax.set(xlabel="Productivity potential, [с.u.]")
    st.pyplot(fig)


def plot_well_cube(
    session_state,
    plot_plan="Yes",
    dot=True,
    corrected_traj=False,
    opacity=0.2,
    cube_to_plot="oil_saturation",
    plot_vis=True,
):
    colors = ["blue", "cyan", "yellow", "orange", "red", "maroon"]
    colorscale = [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]

    if cube_to_plot == "oil_saturation":
        cube = session_state["oil_saturation_file"]
        colorname = "HC saturation [%]"
        colortitle = "Oil saturation cube"
    elif cube_to_plot == "shale":
        cube = session_state["clay_cube"]
        colorname = "Shale content [%]"
        colortitle = "Shale cube"
    elif cube_to_plot == "porosity":
        cube = session_state["porosity_file"]
        colorname = "Porosity [%]"
        colortitle = "Porosity cube"
    elif cube_to_plot == "resistivity":
        cube = session_state["file"]
        colorname = "Resistivity, [ohm*m]"
        colortitle = "Resistivity cube"
    elif cube_to_plot == "permeability":
        cube = session_state["permeability_file"]
        colorname = "Permeability, [md]"
        colortitle = "Permeability"
    elif cube_to_plot == "productivity_potential":
        cube = session_state["prod_potential"]
        colorname = "Productivity potential"
        colortitle = "Productivity potential"

    if cube.shape[0] < cube.shape[2]:
        cube = cube.T

    if plot_plan == "Yes":
        df_planned_well = session_state['planned_traj']


    # Создаем координатные сетки
    x_c, y_c, tvd_c = np.meshgrid(
        np.linspace(session_state['x_cube'], session_state['x_cube'] + cube.shape[0], cube.shape[0] // 5),
        np.linspace(0, 79, cube.shape[1]),
        np.linspace(1990, 2069, cube.shape[2]),
        indexing="ij",
    )

    # Визуализируем куб в 3D
    fig = go.Figure(
        data=go.Volume(
            x=x_c.flatten(),
            y=y_c.flatten(),
            z=tvd_c.flatten(),
            value=cube.flatten(),
            #  isomin=int(data.min()),  # Минимальное значение диапазона значений
            # isomax=int(data.max()),  # Максимальное значение диапазона значений
            opacity=opacity,  # Настраиваем прозрачность для лучшей видимости
            colorscale="jet",  # Настраиваем цветовую схему как у лога сопротивления
            surface_count=3,  # Устанавливаем колличество слоев 3д объекта
            colorbar=dict(lenmode="fraction", len=0.75, thickness=20, title=f"{colorname}"),
        ),
    )
    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=1.5, y=-1.5, z=1),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1),
        )
    )

    if plot_plan == "Yes":
        # well
        scatter = go.Scatter3d(
            x=df_planned_well[:, 0],
            y=df_planned_well[:, 1],
            z=df_planned_well[:, 2],
            mode="lines",
            line=dict(color="black", width=10),
            name="Planned well trajectory",
            opacity=opacity,
        )

        # well
        fig.add_trace(scatter)

        # лейаут
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title="X [m]",
                ),
                # tickvals =[df_planned_well[:, 0][-1]/10 * i for i in range(10)],
                #  ticktext = [f'{int(df_planned_well[:, 0][-1]*ft_to_mt/10 * i)}' for i in range(10)]),
                yaxis=dict(
                    title="Y [m]",
                ),
                # tickvals =[df_planned_well[:, 1][-1]/5 * i for i in range(5)],
                #  ticktext = [f'{int(df_planned_well[:, 1][-1]*ft_to_mt/5 * i)}' for i in range(5)]),
                zaxis=dict(
                    title="TVD [m]",
                    range=[session_state['z_cube'] + 200, session_state['z_cube'] - 100],
                    tickvals=[session_state['z_cube'] - 100 + (300) / 5 * i for i in range(5)],
                    ticktext=[f"{int((session_state['z_cube'] - 100  + (300) ) / 5 * i)}" for i in range(5)],
                ),
                aspectratio=dict(x=2.875, y=0.42, z=1),
            ),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            width=1000,
            height=800,
            title=colortitle,
        )

    if plot_plan == "Yes":
        fig.add_trace(
            go.Scatter3d(
                x=[session_state["X_RTFE"] + session_state['x_cube']],
                y=[session_state["Y_RTFE"] + session_state['y_cube']],
                z=[session_state["Z_RTFE"] + session_state['z_cube']],
                mode="markers",
                line=dict(color="green", width=50),
                name="Start point",
            )
        )


    if plot_vis:
        traj_corr = session_state["traj"].copy()

        st.subheader("Drilling trajectory table")
        traj = np.round(traj_corr, 2)
        if traj[2][-1] < session_state['x_cube']:
            traj[2] += session_state['x_cube']
            traj[1] += session_state["y_cube"]
            traj[0] += session_state['z_cube']

        traj_renewed = traj.copy()
        traj_renewed[0] = traj[2]
        traj_renewed[2] = traj[0]

        # calculate OFV for planned_traj
        x = np.linspace(0, cube.shape[0] - 1, cube.shape[0])
        y = np.linspace(0, cube.shape[1] - 1, cube.shape[1])
        z = np.linspace(0, cube.shape[2] - 1, cube.shape[2])
        points = (x, y, z)
        OFV_planned = 0
        OFV_corr = 0
        c = 0
        for i in range(len(traj_corr[0])):
            x1, y1, z1 = HD_interp(session_state, traj_renewed[0][-i])
            z_cor, y_cor, x_cor = traj_corr[0][-i], traj_corr[1][-i], traj_corr[2][-i]
            try:
                OFV_planned += interpn(
                    points,
                    cube,
                    [x1 - session_state['x_cube'], y1 - session_state["y_cube"], z1 - session_state['z_cube']],
                    method="nearest",
                )
                OFV_corr += interpn(points, cube, [x_cor, y_cor, z_cor], method="nearest")
                c += 1
            except:
                continue

        df = pd.DataFrame(traj_renewed)
        df = df.set_index(pd.Index(["X [m]", "Y [m]", "TVD [m]"]))
        #st.dataframe(df.T)

        fig.add_trace(
            go.Scatter3d(
                x=traj[2],
                y=traj[1],
                z=traj[0],
                mode="lines",
                line=dict(color="blue", width=7),
                name="Corrected well trajectory",
                opacity=1,
            )
        )

        st.subheader("OFV values")

        st.write("Planned trajectory OFV:", round(float(OFV_planned), 2))
        st.write("Corrected trajectory OFV", round(float(OFV_corr), 2))

    st.plotly_chart(fig)




def plot_well_cube_sec(
    session_state,
    plan_well=None,
    plot_plan=False,
    dot=True,
    plot_corrected=False,
    corrected_traj=None,
    opacity=0.2,
    cube_to_plot="oil_saturation",
):
    colors = ["blue", "cyan", "yellow", "orange", "red", "maroon"]
    colorscale = [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]

    if cube_to_plot == "oil_saturation":
        cube = session_state["oil_saturation_file"]
        colorname = "HC saturation [%]"
        colortitle = "Oil saturation cube"
    elif cube_to_plot == "shale":
        cube = session_state["clay_cube"]
        colorname = "Shale content [%]"
        colortitle = "Shale cube"
    elif cube_to_plot == "porosity":
        cube = session_state["porosity_file"]
        colorname = "Porosity [%]"
        colortitle = "Porosity cube"
    elif cube_to_plot == "resistivity":
        cube = session_state["file"]
        colorname = "Resistivity, [ohm*m]"
        colortitle = "Resistivity cube"
    elif cube_to_plot == "permeability":
        cube = session_state["permeability_file"]
        colorname = "Permeability, [md]"
        colortitle = "Permeability"
    elif cube_to_plot == "productivity_potential":
        cube = session_state["prod_potential"]
        colorname = "Productivity potential"
        colortitle = "Productivity potential"

    if cube.shape[0] < cube.shape[2]:
        cube = cube.T
    
    # Создаем координатные сетки
    x_c, y_c, tvd_c = np.meshgrid(
        np.linspace(session_state['x_cube'], session_state['x_cube'] + cube.shape[0], cube.shape[0]),
        np.linspace(session_state['y_cube'], session_state['y_cube'] + cube.shape[1], cube.shape[1]),
        np.linspace(1990, 2069, cube.shape[2]),
        indexing="ij",
    )   

    # Визуализируем куб в 3D
    fig = go.Figure()

    if plot_plan == True:
        # well
        # plot planned well
        df_planned_well = session_state['planned_traj']
        
        scatter = go.Scatter3d(
                x=df_planned_well[:, 0],
                y=df_planned_well[:, 1],
                z=df_planned_well[:, 2],
                mode="lines",
                line=dict(color="green", width=10),
                name="Planned well trajectory",
                opacity=opacity,
            )
        fig.add_trace(scatter)

        
    if dot:
        fig.add_trace(
            go.Scatter3d(
                x=[session_state["X_RTFE"] + session_state['x_cube']],
                y=[session_state["Y_RTFE"] + session_state['y_cube']],
                z=[session_state["Z_RTFE"] + session_state['z_cube']],
                mode="markers",
                line=dict(color="green", width=50),
                name="Start point",
            )
        )
       

    if plot_corrected:
        traj_corr = corrected_traj
        
        traj = np.round(traj_corr, 2)
       
        if traj[2][-1] < session_state['x_cube']:
            traj[2] += session_state['x_cube']
            traj[1] += session_state["y_cube"]
            traj[0] += session_state['z_cube']

        traj_renewed = traj.copy()
        if traj_renewed.shape[0] > 10:
            traj_renewed = traj_smoothing(traj_renewed, 5)
            traj_renewed[:5] = traj[:5]
        traj_renewed = traj_renewed.T
        
        
        # df = pd.DataFrame(traj_renewed)
        # df = df.columns((["X [m]", "Y [m]", "TVD [m]"]))

        fig.add_trace(
            go.Scatter3d(
                x=traj_renewed[:, 2],
                y=traj_renewed[:, 1],
                z=traj_renewed[:, 0],
                mode="lines",
                line=dict(color="black", width=8),
                name="Corrected well trajectory",
                opacity=1,
            )
        )

    for i in range(0, len(corrected_traj[0] - 1)):
        x_index = int(corrected_traj[0][i])
        y_index = int(corrected_traj[1][i])
        z_index = int(corrected_traj[2][i])
        sec = cube[z_index, :, :]
        if i == 0:
            showscale = True
        else:
            showscale = False
        #st.write(x_index, y_index, z_index)
        fig.add_trace(
            go.Surface(
                x=x_c[z_index, :, :],
                y=y_c[y_index, :, :],
                z=tvd_c[x_index, :, :],
                surfacecolor=sec,
                opacity=opacity,
                colorscale=colorscale,
                showscale=showscale,
                contours={
                    "x": {"show": True},
                    "y": {"show": True},
                    "z": {"show": True},
                },
                name=f"Step {i}",
                colorbar=dict(
                    title=colorname,
                    x=1.1,  # Adjust the x position of the colorscale (0-1 range)
                    y=0.5,  # Adjust the y position of the colorscale (0-1 range)
                    len=0.75,  # Adjust the length of the colorscale (0-1 range)
                ),
            )
        )
     
    fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title="X [ft]",
                    range=[session_state['x_cube'] + 200, session_state['x_cube'] - 100],
                ),
                # tickvals =[df_planned_well[:, 0][-1]/10 * i for i in range(10)],
                #  ticktext = [f'{int(df_planned_well[:, 0][-1]*ft_to_mt/10 * i)}' for i in range(10)]),
                yaxis=dict(
                    title="Y [ft]",
                ),
                # tickvals =[df_planned_well[:, 1][-1]/5 * i for i in range(5)],
                #  ticktext = [f'{int(df_planned_well[:, 1][-1]*ft_to_mt/5 * i)}' for i in range(5)]),
                zaxis=dict(
                    title="TVD [ft]",
                    range=[session_state['z_cube'] + 200, session_state['z_cube'] - 100],
                    tickvals=[session_state['z_cube'] - 100 + (300) / 5 * i for i in range(5)],
                   # ticktext=[f"{int((session_state['z_cube'] - 100  + (300) ) / 5 * i)}" for i in range(5)],
                ),
                aspectratio=dict(x=1.5, y=0.8, z=1.5), 
                


            ),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            width=1000,
            height=800,
            title=colortitle,
        )

    st.plotly_chart(fig)


# лейаут
        # fig.update_layout(
        #     scene=dict(
        #         xaxis=dict(
        #             title="X [m]",
        #         ),
        #         yaxis=dict(
        #             title="Y [m]",
        #         ),
        #         zaxis=dict(title="TVD [m]"),
        #         aspectratio=dict(x=2.875, y=0.42, z=1),
        #     ),
        #     legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        #     width=400,
        #     height=100,
        #     title=colortitle,
        # )


def add_trace(fig, cube, session_state, traj_corr):
    x_c, y_c, tvd_c = np.meshgrid(
        np.linspace(370, 599, cube.shape[0]),
        np.linspace(0, 79, cube.shape[1]),
        np.linspace(1990, 2069, cube.shape[2]),
        indexing="ij",
    )
    x_index_2 = 140
    section_x_2 = cube[x_index_2, :, :]

    fig.add_trace(
        go.Surface(
            x=x_c[int(traj_corr[2][-1]), :, :],
            y=y_c[int(traj_corr[2][-1]), :, :],
            z=tvd_c[int(traj_corr[2][-1]), :, :],
            surfacecolor=section_x_2,
            opacity=0.3,
            # colorscale='jet',
            # showscale=True,
            contours={
                "x": {"show": False},
                "y": {"show": False},
                "z": {"show": False},
            },
            name=f"Step {len(traj_corr)}",
        )
    )
    return fig
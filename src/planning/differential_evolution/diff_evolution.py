import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import pickle
from scipy.interpolate import interpn
import pandas as pd
import plotly.express as px
from plotly.offline import plot
from plotly.subplots import make_subplots


class DE_algo:
    def __init__(self, cube_3d):
        # make padding for new cube
        new_cube = np.zeros(shape=(cube_3d.shape[0] + 40, cube_3d.shape[1] + 40, cube_3d.shape[2] + 40))
        new_cube[: cube_3d.shape[0], : cube_3d.shape[1], : cube_3d.shape[2]] = cube_3d
        self.cube_3d = new_cube
        self.old_cube = cube_3d
        x = np.linspace(0, new_cube.shape[0] - 1, new_cube.shape[0])
        y = np.linspace(0, new_cube.shape[1] - 1, new_cube.shape[1])
        z = np.linspace(0, new_cube.shape[2] - 1, new_cube.shape[2])
        self.points = (x, y, z)

    def get_vec(self, inc, azi, length, nev=False, deg=True):
        """
        Convert inc and azi into a vector.
        Params:
            inc: array of n floats
                Inclination relative to the z-axis (up)
            azi: array of n floats
                Azimuth relative to the y-axis
            r: float or array of n floats
                Scalar to return a scaled vector
        Returns:
            An (n,3) array of vectors
        """
        if deg:
            inc_rad, azi_rad = np.radians(np.array([inc, azi]))
        else:
            inc_rad = inc
            azi_rad = azi
        y = length * np.sin(inc_rad) * np.cos(azi_rad)
        x = length * np.sin(inc_rad) * np.sin(azi_rad)
        z = length * np.cos(inc_rad)

        #     if nev:
        #         vec = np.array([y, x, z]).T
        #     else:
        #         vec = np.array([x, y, z]).T
        return np.stack([x, y, z])

    def obj(self, angles, *state):
        """
        Сalculate objective function on 1 step
        Params:
                angles: list of two values: inclination and azimuth
                state: list of values: [x,y,z,list of inclinations, list of azimuth,
                angle constraint, length of 1 step]

        Returns:
                An float normalized value of OFV
        """

        penalty = 0
        vec_diff = self.get_vec(angles[0], angles[1], state[-1])

        state_new = [
            state[0][0] + vec_diff[0],
            state[0][1] + vec_diff[1],
            state[0][2] + vec_diff[2],
        ]

        # dogleg severity constraint
        dls_val = np.linalg.norm(angles[0] - state[1][-2]) + np.linalg.norm(angles[1] - state[2][-2])
        if dls_val >= state[3] * state[4]:
            penalty += dls_val * 0.5

        # contraint for high length action
        length_constraint = np.linalg.norm(vec_diff)

        # objective function
        OFV = interpn(self.points, self.cube_3d, state_new, method="nearest") * 100 / length_constraint - penalty

        OFV = OFV  # + length_rew*1.5

        return -OFV

    def obj2(self, angles, *state):
        """
        Сalculate objective function on 3 step ahead
            Params:
                angles: list of six values: first three for inclination, next three for azimuth
                state: list of values: [x,y,z,list of inclinations, list of azimuth,
                angle constraint, length of 1 step]

            Returns:
                An float normalized value of OFV
        """
        OFV = 0
        penalty = 0

        azi_l = [state[2][-1]]
        incl_l = [state[1][-1]]
        state_new = [state[0][0], state[0][1], state[0][2]]

        if state_new[2] + 3 * state[4] <= self.cube_3d.shape[2]:
            vec_diff = self.get_vec(angles[0], angles[3], state[-1])
            state_new = [
                state_new[0] + vec_diff[0],
                state_new[1] + vec_diff[1],
                state_new[2] + vec_diff[2],
            ]
            azi_l.append(angles[0])
            incl_l.append(angles[3])
            dls_val = np.linalg.norm(angles[0] - azi_l[-1]) + np.linalg.norm(angles[3] - incl_l[-1])
            if dls_val >= state[3] * state[4]:
                penalty += dls_val * 0.5

            # contraint for high length action
            length_constraint = np.linalg.norm(vec_diff)

            idx_max = np.argmax(self.cube_3d.shape)
            length_rew = 1 - (self.cube_3d.shape[idx_max] - state_new[idx_max] / self.cube_3d.shape[idx_max])
            #  length_rew = state_new[idx_max] - state[0][idx_max]

            # objective function

            OFV += interpn(self.points, self.cube_3d, state_new, method="nearest") / length_constraint - penalty
            OFV += length_rew * 5

        else:
            for i in range(1):
                vec_diff = self.get_vec(angles[i], angles[3 + i], state[-1])

                state_new = [
                    state_new[0] + vec_diff[0],
                    state_new[1] + vec_diff[1],
                    state_new[2] + vec_diff[2],
                ]

                azi_l.append(angles[i])
                incl_l.append(angles[3 + i])
                # dogleg severity constraint
                dls_val = np.linalg.norm(angles[i] - azi_l[-1]) + np.linalg.norm(angles[3 + i] - incl_l[-1])
                if dls_val >= state[3] * state[4]:
                    penalty += dls_val * 0.5

                # constraint for high length action
                length_constraint = np.linalg.norm(vec_diff)

                idx_max = np.argmax(self.cube_3d.shape)
                length_rew = 1 - (self.cube_3d.shape[idx_max] - state_new[idx_max] / self.cube_3d.shape[idx_max])

                # objective function
                OFV += interpn(self.points, self.cube_3d, state_new, method="nearest") / length_constraint - penalty
                OFV += length_rew * 5

        return -OFV

    def calculate_OFV(self, state):
        """
        Get oil saturation value for current step
           Params:
               state: list of values: [x,y,z,list of inclinations, list of azimuth,
               angle constraint, length of 1 step]

           Returns:
                   Float value of OFV
        """
        OFV = interpn(self.points, self.cube_3d, state, method="nearest")
        return OFV

    def check_edge_limitation(self, state):
        edge_reached = False

        if (
            state[0] >= self.old_cube.shape[0]
            or state[1] >= self.old_cube.shape[1]
            or state[2] >= self.old_cube.shape[2]
        ):
            edge_reached = True

        return edge_reached

    def DE_step(
        self,
        OFV=0,
        azi_l=None,
        incl_l=None,
        traj=None,
        init_pos=None,
        pop_size=100,
        maxiter=1000,
        F=0.7,
        bounds=[(0, 180), (0, 180), (0, 180), (0, 92), (0, 92), (0, 92)],
        length=10,
        angle_constraint=10,
        strategy="best1bin",
    ):
        """
        Make 1 optimization step in property cube by using differential evolution algorithm.
        ______________
        Params:
            :param OFV: float
                current OFV value
            :param azi_l: list
                list of azimuth values for all previous steps
            :param incl_l:
                list of inclinations values for all previous steps
            :param traj: np.array
                array of the current trajectory
            :param init_pos: list
                list of initial coordinates if the step is initial for the trajectory
            :param pop_size: int
                population size parameter for differential evolution
            :param maxiter:  int
                maximum number of iterations for differential evolution
            :param F: float (0,1)
            mutation parameter for differential evolution
            :param bounds: list
            list of tuples of angle bounds. first threea tuples are for azimuth angles,
             last three for zenith
            :param length: int
            step length
            :param angle_constraint: list
            predefined angle constraint for the trajectory
            :param strategy: str
             strategy parameter for differential evolution
        :return:
            OFV, trajectory, azimuth_l, inclination_l
            new OFV value, updated trajectory, list of azimuth and zenith values
        """

        if init_pos != None:
            azi_l = [0, 0]
            incl_l = [0, 0]
            state = (
                [init_pos[0], init_pos[1], init_pos[2]],
                incl_l,
                azi_l,
                angle_constraint,
                length,
            )
            traj_x = [init_pos[0]]
            traj_y = [init_pos[1]]
            traj_z = [init_pos[2]]
        else:
            traj_x = list(traj[0])
            traj_y = list(traj[1])
            traj_z = list(traj[2])
            state = (
                [traj[0][-1], traj[1][-1], traj[2][-1]],
                incl_l,
                azi_l,
                angle_constraint,
                length,
            )
        self.state = state
        de_sol = differential_evolution(
            self.obj2,
            bounds,
            args=(state),
            mutation=F,
            popsize=pop_size,
            strategy=strategy,
            maxiter=maxiter,
            updating="deferred",
            disp=False,
        ).x

        incl_l.append(de_sol[0])
        azi_l.append(de_sol[1])
        step = self.get_vec(incl_l[-1], azi_l[-1], length=length)

        traj_x.append(state[0][0] + step[0])
        traj_y.append(state[0][1] + step[1])
        traj_z.append(state[0][2] + step[2])

        state[0][0] = state[0][0] + step[0]
        state[0][1] = state[0][1] + step[1]
        state[0][2] = state[0][2] + step[2]
        state_new = [state[0][0], state[0][1], state[0][2]]

        edge_limitation = self.check_edge_limitation(state_new)

        OFV += interpn(self.points, self.cube_3d, state_new, method="nearest")

        return OFV, np.stack([traj_x, traj_y, traj_z]), azi_l, incl_l, edge_limitation

    def DE_planning(
        self,
        pop_size=100,
        maxiter=1000,
        F=0.7,
        cr=0.7,
        strategy="best1bin",
        bounds=[(0, 180), (0, 180), (0, 180), (0, 92), (0, 92), (0, 92)],
        length=10,
        angle_constraint=0.1,
        init_incl=[0, 0],
        init_azi=[10, 10],
        init_pos=[30, 30, 150],
    ):
        """
        1 step differential evolution trajectory planning.
        Params:
            pop_size: int
             population size
            num_iters: int
             define number of iterations
            F: float (0,1)
             scale factor for mutation
            cr: flaot (0,1)
             crossover rate for recombination
            bounds: list of tuples
             bound for searchable paramaters (in our case (azi, inclination))
            angle_constraint: float
             dogleg constraint per m
            length: int
             length of one step
        :return:
             OFV, trajectory, azimuth_l, inclination_l
            new OFV value, updated trajectory, list of azimuth and zenith values

        """
        OFV = 0
        pos = init_pos
        incl_l = init_incl
        azi_l = init_azi
        state = ([pos[0], pos[1], pos[2]], incl_l, azi_l, angle_constraint, length)
        traj_x = [state[0][0]]
        traj_y = [state[0][1]]
        traj_z = [state[0][2]]
        self.state = state
        c = 0
        dic = {"Step": [], "X": [], "Y": [], "Z": []}
        while (
            (traj_z[-1] <= self.old_cube.shape[2] - length)
            and (traj_y[-1] <= self.old_cube.shape[1] - length)
            and (traj_x[-1] <= self.old_cube.shape[0] - length)
        ):
            de_sol = differential_evolution(
                self.obj,
                bounds,
                strategy=strategy,
                args=(state),
                mutation=F,
                popsize=pop_size,
                maxiter=maxiter,
                updating="deferred",
                disp=False,
            ).x
            incl_l.append(de_sol[0])
            azi_l.append(de_sol[1])
            step = self.get_vec(incl_l[-1], azi_l[-1], length=length)

            traj_x.append(state[0][0] + step[0])
            traj_y.append(state[0][1] + step[1])
            traj_z.append(state[0][2] + step[2])

            state[0][0] = state[0][0] + step[0]
            state[0][1] = state[0][1] + step[1]
            state[0][2] = state[0][2] + step[2]
            state_new = [state[0][0], state[0][1], state[0][2]]

            edge_limitation = self.check_edge_limitation(state_new)

            print(f"Step - {c}, x : {round(traj_x[-1], 3)}, y: {round(traj_y[-1], 3)}, z: {round(traj_z[-1], 3)}")
            dic["Step"].append(c)
            dic["X"].append(round(traj_x[-1], 3))
            dic["Y"].append(round(traj_y[-1], 3))
            dic["Z"].append(round(traj_z[-1], 3))
            OFV += interpn(self.points, self.cube_3d, state_new, method="nearest")
            c += 1
        print(OFV)
        df = pd.DataFrame(dic)
        return (
            OFV,
            np.stack([traj_x, traj_y, traj_z]),
            df,
            incl_l,
            azi_l,
            edge_limitation,
        )


def plot_results(volume_cube, traj_corr, traj_plan=None):
    x = []
    z = []
    y = []

    property_along_y = []
    property_along_x = []
    property_along_z = []
    traj_x, traj_y, traj_z = traj_corr
    if traj_plan != None:
        traj_x_plan, traj_y_plan, traj_z_plan = traj_plan
    for i in range(0, len(traj_x)):
        x.append(traj_x[i])
        z.append(traj_z[i])
        y.append(traj_y[i])
        property_along_y.append(volume_cube[round(traj_x[i]), :, round(traj_z[i])].T)
        property_along_x.append(volume_cube[:, round(traj_y[i]), round(traj_z[i])].T)
        property_along_z.append(volume_cube[round(traj_x[i]), round(traj_y[i]), :].T)

    x_t = np.array(x)
    z_t = np.array(z)
    y_t = np.array(y)

    fig, ax = plt.subplots(3, 1, figsize=(15, 10))
    property_along_y_arr = np.array(property_along_y)
    property_along_x_arr = np.array(property_along_x)
    property_along_z_arr = np.array(property_along_z)

    mean_x = round(np.array(traj_x).mean())
    mean_y = round(np.array(traj_y).mean())
    mean_z = round(np.array(traj_z).mean())

    ax[0].plot(z_t, y_t, color="r", linewidth=3, label="Corrected")
    ax[0].scatter(z_t, y_t, color="black", linewidth=1, marker="|", label="Iteration steps")
    for i in range(0, len(z_t)):
        ax[0].annotate(f"{i + 1}", (z[i], y[i]))

    ax[0].set_title("XZ trajectory projection")
    ax[0].imshow(volume_cube[mean_x, :, :])
    ax[0].legend(loc="upper right")

    ax[1].plot(z_t, x_t, color="r", linewidth=3, label="Corrected")
    ax[1].set_title("YZ trajectory projection")
    ax[1].imshow(volume_cube[:, mean_y, :])
    ax[1].legend(loc="upper right")

    ax[2].set_title("XY trajectory projection")

    ax[2].imshow(volume_cube[:, :, mean_z])
    ax[2].plot(x_t, y_t, color="r", linewidth=3, label="Corrected")
    ax[2].legend(loc="upper right")
    if traj_plan != None:
        ax[0].plot(traj_z_plan, traj_y_plan, color="blue", linewidth=1, label="Planned")
        ax[2].plot(traj_x_plan, traj_y_plan, color="blue", linewidth=1, label="Planned")
        ax[1].plot(traj_z_plan, traj_x_plan, color="blue", linewidth=1, label="Planned")
    return fig


def plot_results_2(
    volume_cube,
    traj_corr,
    traj_plan=None,
    x_coords=2000,
    y_coords=500,
    z_coords=500,
    data="oil_sat",
):
    """
    Plot results using matplotlib:
    Params:
        volume_cube: 3d array
            numpy array of oil saturation cube
        traj_corr: list
            list of x,y,z coordinates
        traj_plan: list
            planned trajectory
        x_coords, y_coords, z_coords: int
            initial coordinates for cube

    :return:
        matplotlib plot of mean projections on 3 planes with trajectories

    """

    x = []
    z = []
    y = []

    if data == "oil_sat":
        colorname = "HC saturation, frac"
    elif data == "resistivity":
        colorname = "Resitivity, Ohm*m"

    property_along_y = []
    property_along_x = []
    property_along_z = []
    traj_x, traj_y, traj_z = traj_corr
    if traj_plan is not None:
        traj_x_plan, traj_y_plan, traj_z_plan = traj_plan[0], traj_plan[1], traj_plan[2]
    for i in range(0, len(traj_x)):
        x.append(traj_x[i])
        z.append(traj_z[i])
        y.append(traj_y[i])
        property_along_y.append(volume_cube[round(traj_x[i]), :, round(traj_z[i])].T)
        property_along_x.append(volume_cube[:, round(traj_y[i]), round(traj_z[i])].T)
        property_along_z.append(volume_cube[round(traj_x[i]), round(traj_y[i]), :].T)

    x_t = np.array(x) + x_coords
    z_t = np.array(z) + z_coords
    y_t = np.array(y) + y_coords

    x = np.linspace(x_coords, x_coords + volume_cube.shape[0], volume_cube.shape[0])
    y = np.linspace(y_coords, y_coords + volume_cube.shape[1], volume_cube.shape[1])
    z = np.linspace(z_coords, z_coords + volume_cube.shape[2], volume_cube.shape[2])

    mean_x = round(np.array(traj_x).mean())
    mean_y = round(np.array(traj_y).mean())
    mean_z = round(np.array(traj_z).mean())
    if len(traj_z) > 2:
        step = traj_z[-1] - traj_z[-2]
    else:
        step = 1

    fig = px.imshow(
        volume_cube[int(traj_x[-1]), :, :],
        color_continuous_scale="rainbow",
        origin="lower",
        aspect="auto",
        labels={f"x": "X, m", "y": "Y, m ", "color": colorname},
        x=z,
        y=y,
    )

    fig.update_xaxes(showticklabels=True)
    fig.add_scatter(
        x=z_t,
        y=y_t,
        name="Corrected",
        marker_color="black",
        mode="markers+text",
        showlegend=False,
        text=[f"{i}" for i in range(len(traj_corr[0]))],
        textposition="top center",
        marker=dict(size=8, symbol="line-ns", line=dict(width=2, color="DarkSlateGrey")),
    )
    fig.update_layout(
        font=dict(
            #  family="Courier New, monospace",
            size=18,  # Set the font size here
            color="Black",
        ),
    )
    fig.update_layout(legend=dict(x=0.01, y=1))
    fig.update_layout(
        height=400,
        width=1300,
        title_text="Plan view",
        # title_loc ='center'
    )
    fig.add_scatter(
        x=z_t,
        y=y_t,
        name="Corrected",
        line=dict(width=4),
        marker_color="black",
        mode="lines",
        showlegend=True,
    )

    fig2_1 = px.imshow(
        volume_cube[:, :, int(traj_z[-1])],
        color_continuous_scale="rainbow",
        origin="lower",
        aspect="auto",
        labels={f"x": "Y, m", "y": "TVD, m "},
        x=x,
        y=y,
    )
    fig2_1.update_yaxes(autorange="reversed")

    fig2_1.update_xaxes(showticklabels=True)
    fig2_1.update_layout(
        font=dict(size=18, color="Black"),  # Set the font size here
        legend=dict(x=0.01, y=1),
        title_text="Current step",
    )

    fig2_1.add_scatter(
        x=[x_t[-1]],
        y=[y_t[-1]],
        name="Corrected",
        line=dict(width=300),
        marker_color="black",
        mode="markers",
        showlegend=True,
    )

    if traj_plan is not None:
        fig2_1.add_scatter(
            x=[traj_x_plan[-1]],
            y=[traj_y_plan[-1]],
            name="Planned",
            line=dict(width=300),
            marker_color="#8c564b",
            mode="markers",
            showlegend=True,
        )

    if traj_z[-1] + step > volume_cube.shape[2]:
        step = 0
    fig2_2 = px.imshow(
        volume_cube[:, :, int(traj_z[-1] + step)],
        color_continuous_scale="rainbow",
        origin="lower",
        aspect="auto",
        labels={f"x": "Y, m", "y": "TVD, m ", "color": colorname},
        x=x,
        y=y,
    )

    fig2_2.update_xaxes(showticklabels=True)

    fig2_2.update_layout(
        font=dict(size=18, color="Black"),  # Set the font size here
    )
    fig2_2.update_yaxes(autorange="reversed")
    fig2_2.update_layout(legend=dict(x=0.01, y=1))
    fig2_2.update_layout(
        title_text="Next step step",
        # title_loc ='center'
    )

    fig3 = px.imshow(
        volume_cube[:, int(traj_y[-1]), :],
        color_continuous_scale="rainbow",
        origin="lower",
        aspect="auto",
        labels={f"x": "X, m", "y": "TVD, m ", "color": colorname},
        x=z,
        y=x,
    )

    fig3.update_xaxes(showticklabels=True)

    fig3.update_layout(
        font=dict(
            #  family="Courier New, monospace",
            size=18,  # Set the font size here
            color="Black",
        ),
        height=400,
        width=1300,
        title_text="Long section view",
        legend=dict(x=0.01, y=1),
    )

    fig3.add_scatter(
        x=z_t,
        y=x_t,
        name="Corrected",
        marker_color="black",
        line=dict(width=4),
        mode="lines+text",
        text=[f"{i}" for i in range(len(traj_corr[0]))],
        showlegend=True,
    )

    return fig, fig3, fig2_1, fig2_2


def plot_results_2d(
    cross_sec,
    traj_corr,
    traj_plain=None,
    colorname="Resistivity, Ohm*m",
    colormap="Reds",
):
    #   x = np.linspace(0, 0 + cube_3d.shape[0], cube_3d.shape[0])
    y = np.linspace(0, 0 + cross_sec.shape[0], cross_sec.shape[0])
    z = np.linspace(0, 0 + cross_sec.shape[1], cross_sec.shape[1])

    fig = px.imshow(
        cross_sec,
        color_continuous_scale=colormap,
        origin="lower",
        aspect="auto",
        labels={f"x": "X, m", "y": "Y, m ", "color": colorname},
        x=z,
        y=y,
    )
    fig.update_xaxes(showticklabels=True)

    fig.update_layout(
        font=dict(
            #  family="Courier New, monospace",
            size=18,  # Set the font size here
            color="Black",
        ),
    )
    fig.update_layout(legend=dict(x=0.01, y=1))
    fig.update_layout(
        height=400,
        width=1300,
        title_text="Plan view",
        # title_loc ='center'
    )

    if traj_plain != None:
        fig.add_scatter(
            x=traj_plain[0],
            y=traj_plain[1],
            name="Planned",
            marker_color="black",
            mode="lines",
            showlegend=True,
            text=[f"{i}" for i in range(len(traj_corr[0]))],
            textposition="top center",
            marker=dict(size=8, symbol="line-ns", line=dict(width=2, color="DarkSlateGrey")),
        )

    fig.add_scatter(
        x=traj_corr[0],
        y=traj_corr[1],
        name="Corrected",
        marker_color="Blue",
        mode="lines",
        showlegend=True,
        text=[f"{i}" for i in range(len(traj_corr[0]))],
        textposition="top center",
        marker=dict(size=8, symbol="line-ns", line=dict(width=2, color="Blue")),
    )
    fig.update_layout(legend=dict(x=0.01, y=1))
    return fig

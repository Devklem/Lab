import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import pickle
from scipy.interpolate import interpn
import pandas as pd
import plotly.express as px
from plotly.offline import plot
from plotly.subplots import make_subplots
from stochopy.optimize import minimize


class RTFE_main:
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

    def obj(self, angles):
        """
        Сalculate objective function on 1 step
        Params:
                angles: list of two values: inclination and azimuth
                state: list of values: [x,y,z,list of inclinations, list of azimuth,
                angle constraint, length of 1 step]

        Returns:
                An float normalized value of OFV
        """
        state = self.state
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

    def obj2(self, angles):
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
        state = self.state
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
                penalty += dls_val * 1

            # contraint for high length action
            length_constraint = np.linalg.norm(vec_diff)

            idx_max = np.argmax(self.cube_3d.shape)
            #     length_rew = 1 - (self.cube_3d.shape[idx_max] - state_new[idx_max]) / self.cube_3d.shape[idx_max]

            length_rew = state_new[idx_max] - state[0][idx_max]

            # objective function

            OFV += interpn(self.points, self.cube_3d, state_new, method="nearest") * 1000 / length_constraint - penalty
            OFV += length_rew * 10

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
                    penalty += dls_val * 2

                # constraint for high length action
                length_constraint = np.linalg.norm(vec_diff)

                idx_max = np.argmax(self.cube_3d.shape)
                length_rew = 1 - (self.cube_3d.shape[idx_max] - state_new[idx_max] / self.cube_3d.shape[idx_max])

                # objective function
                OFV += interpn(self.points, self.cube_3d, state_new, method="nearest") * 1000 / length_constraint - penalty
                OFV += length_rew * 10

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
        config,
        OFV=0,
        azi_l=None,
        incl_l=None,
        traj=None,
        init_pos=None,
        bounds=[(0, 180), (0, 180), (0, 180), (0, 92), (0, 92), (0, 92)],
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
        angle_constraint = config["angle_constraint"]
        length = config["length"]
        F = config["F"]

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
        if config["solver"] == "DE_scipy":
            de_sol = differential_evolution(
                self.obj2,
                bounds,
                strategy=config["strategy"],
                args=(state),
                mutation=F,
                popsize=config["pop_size"],
                maxiter=config["maxiter"],
                updating="deferred",
                disp=False,
            ).x

        else:
            de_sol = minimize(
                self.obj2,
                bounds,
                method=config["solver"],
                options={
                    "maxiter": config["maxiter"],
                    "popsize": config["pop_size"],
                    "seed": 0,
                },
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

        self.state = state

        OFV += interpn(self.points, self.cube_3d, state_new, method="nearest")

        return OFV, np.stack([traj_x, traj_y, traj_z]), azi_l, incl_l, edge_limitation

    def get_next_step(self, state, incl, azi, length):
        step = self.get_vec(incl, azi, length=length)

        if len(self.traj_z) > 2 and (state[0][2] + step[2] - self.traj_z[-1] < length / 3):
            step = self.get_vec(0, azi, length=length)
            

        self.traj_x.append(state[0][0] + step[0])
        self.traj_y.append(state[0][1] + step[1])
        self.traj_z.append(state[0][2] + step[2])

        state[0][0] = state[0][0] + step[0]
        state[0][1] = state[0][1] + step[1]
        state[0][2] = state[0][2] + step[2]

        edge_limitation = self.check_edge_limitation(state[0])

        return state, edge_limitation


    def DE_planning(
        self,
        config,
        bounds=[(0, 180), (0, 180), (0, 180), (0, 92), (0, 92), (0, 92)],
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
            cr: float (0,1)
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
        angle_constraint = config["angle_constraint"]
        length = config["length"]
        pos = init_pos
        incl_l = init_incl
        azi_l = init_azi
        state = ([pos[0], pos[1], pos[2]], incl_l, azi_l, angle_constraint, length)
        self.traj_x = [state[0][0]]
        self.traj_y = [state[0][1]]
        self.traj_z = [state[0][2]]
        c = 0
        dic = {"Step": [], "X": [], "Y": [], "Z": []}
        while (
            (self.traj_z[-1] <= self.old_cube.shape[2] - length)
            and (self.traj_y[-1] <= self.old_cube.shape[1] - length)
            and (self.traj_x[-1] <= self.old_cube.shape[0] - length)
        ):
            self.state = state

            if config["solver"] == "DE_scipy":
                de_sol = differential_evolution(
                    self.obj2,
                    bounds,
                    strategy=config["strategy"],
                    mutation=config["F"],
                    popsize=config["pop_size"],
                    maxiter=config["maxiter"],
                    updating="deferred",
                    disp=False,
                ).x

            else:
                de_sol = minimize(
                    self.obj2,
                    bounds,
                    method=config["solver"],
                    options={
                        "maxiter": config["maxiter"],
                        "popsize": config["pop_size"],
                        "seed": 0,
                    },
                ).x

            incl_l.append(de_sol[0])
            azi_l.append(de_sol[3])
            # step = self.get_vec(incl_l[-1], azi_l[-1], length=length)

            state, edge_limitation = self.get_next_step(state, incl_l[-1], azi_l[-1], length)

            state_new = [state[0][0], state[0][1], state[0][2]]

            self.state = state

            print(
                f"Step - {c}, x : {round(self.traj_x[-1], 3)}, y: {round(self.traj_y[-1], 3)}, z: {round(self.traj_z[-1], 3)}"
            )
            dic["Step"].append(c)
            dic["X"].append(round(self.traj_x[-1], 3))
            dic["Y"].append(round(self.traj_y[-1], 3))
            dic["Z"].append(round(self.traj_z[-1], 3))
            OFV += interpn(self.points, self.cube_3d, state_new, method="nearest")
            c += 1
        print(OFV)
        df = pd.DataFrame(dic)
        return (
            OFV,
            np.stack([self.traj_x, self.traj_y, self.traj_z]),
            df,
            incl_l,
            azi_l,
            edge_limitation,
        )

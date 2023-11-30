# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 12:39:42 2022

@authors: georgy.peshkov, Maksimilian Pavlov
"""
import streamlit as st
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from src.utils import create_cache_folder
from src import load_data_section, guide_section, oil_sat_calc, \
RTFE_parameters_set_section, simulation_section, data_vis_section
import seaborn as sns
sns.set(rc={'axes.facecolor': 'f4f1ea', 'figure.facecolor': 'f4f1ea'})

if 'button_clicked_data' not in st.session_state:
    st.session_state['button_clicked_data'] = False
    st.session_state['button_clicked'] = False
    st.session_state['button_clicked'] = False
    st.session_state['button_set_clicked'] = False
    st.session_state['button_launch_clicked'] = False
    st.session_state['full_planning_clicke'] = True
    st.session_state['button_info_clicked'] = False
   #st.session_state['traj'] = None
    st.session_state['step_planning_clicked'] = False
    st.session_state['button_clay_cube_clicked'] = False
    st.session_state['button_oil_sat_clicked'] = False
    st.session_state.disabled = False

if 'disabled' not in st.session_state:
    st.session_state.disabled = False
if 'traj_ready' not in st.session_state:
    st.session_state.traj_ready = False


#####################################################################################################################
folder_path_res = str(Path("data/raw/res"))
folder_path_por = str(Path("data/raw/por"))
folder_path_traj = str(Path("data/raw/well"))
st.set_page_config(layout="wide")
create_cache_folder()
st.markdown("<h1 style='text-align: center; color: black;'> Real time formation evaluation </h1>", unsafe_allow_html=True)

##################################################################################################################### 
# Input data subheader
load_data_section(st.session_state)

#####################################################################################################################
# Oil saturation calculation subheader
oil_sat_calc(st.session_state)

#####################################################################################################################

# RTFE set parameters section
RTFE_parameters_set_section(st.session_state)

#####################################################################################################################
# Data visualization section
data_vis_section(st.session_state)

##################################################################################################################### 
# Simulation section
simulation_section(st.session_state)

#####################################################################################################################
# Guidance section
guide_section()

from src.visualization import plot_well_cube
## check plot
import pickle
from src.utils import save_uploadedfile, upload_selected_file, file_selector
from src.states import callback_data
from src.unittests import check_types, check_well_types


def callback_test():
    st.session_state.button_clicked_data = False
    st.session_state.button_clicked = False
    st.session_state.button_set_clicked = False
    st.session_state.button_launch_clicked = False
    st.session_state.button_info_clicked = False
    st.session_state.button_oil_sat_clicked = False
    st.session_state.test = True


def callback_test2():
    st.session_state['test'] = True
    st.session_state['step_planning_clicked2'] = True
#
# st.sidebar.button('Test', on_click=callback_test)
#
# if st.session_state.test:
#     st.session_state['step_planning_clicked2'] = st.button('Step planning test',
#                                                     on_click=callback_test2, key = 'preserve')
#     with open('data/raw/well/Planned_well.pickle','rb') as f:
#         file = pickle.load(f)
#
#
#     st.write(st.session_state.test,st.session_state.step_planning_clicked2)
#     st.session_state['planned_traj'] = check_well_types(file)
#
#     if st.session_state['step_planning_clicked2']:
#             cube_to_plot = st.selectbox('Choose cube to plot',
#                                         ('Resistivity', 'Porosity', 'Oil saturation', 'Shale'),
#                                         on_change = callback_test2)
#             op_slider = st.slider("Choose the opacity", min_value=0.1, max_value=1.0,
#                                 on_change = callback_test2)
#
#             # plot_well_cube(st.session_state, st.session_state['planned_traj'], dot=False
#             #                , corrected_traj=False)
#
#     else:
#             st.sidebar.write('')
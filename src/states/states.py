import streamlit as st

"""
States definition for each button on a streamlit subheader

"""


# session state for load button
def callback_data():
    st.session_state.button_clicked_data = True
    st.session_state.button_clicked = False
    st.session_state.type_of_load = "Select from existing file"
    st.session_state.button_set_clicked = False
    st.session_state.button_launch_clicked = False
    st.session_state.button_info_clicked = False
    st.session_state.button_oil_sat_clicked = False
    st.session_state.button_prod_potential = False
    st.session_state.button_wd_clicked = False


# session state for data visualization button
def callback():
    st.session_state.button_clicked_data = False
    st.session_state.button_clicked = True
    st.session_state.button_set_clicked = False
    st.session_state.button_launch_clicked = False
    st.session_state.button_info_clicked = False
    st.session_state.button_oil_sat_clicked = False
    st.session_state.button_prod_potential = False
    st.session_state.button_wd_clicked = False


# session state for "RFTE parameters" button
def callback_set():
    st.session_state.button_clicked_data = False
    st.session_state.button_clicked = False
    st.session_state.button_set_clicked = True
    st.session_state.button_launch_clicked = False
    st.session_state.button_info_clicked = False
    st.session_state.button_oil_sat_clicked = False
    st.session_state.button_prod_potential = False
    st.session_state.button_wd_clicked = False


# session state for calculation button
def callback_launch():
    st.session_state.button_clicked_data = False
    st.session_state.button_clicked = False
    st.session_state.button_set_clicked = False
    st.session_state.button_launch_clicked = True
    st.session_state.button_info_clicked = False
    st.session_state.button_oil_sat_clicked = False
    st.session_state.button_prod_potential = False
    st.session_state.button_wd_clicked = False


# session state for calculation button
def callback_full_planning():
    st.session_state.button_clicked_data = False
    st.session_state.button_clicked = False
    st.session_state.button_set_clicked = False
    st.session_state.button_launch_clicked = True
    st.session_state.button_info_clicked = False
    st.session_state.button_oil_sat_clicked = False
    st.session_state.full_planning_clicked = True
    st.session_state.button_prod_potential = False
    st.session_state.button_wd_clicked = False


# session state for step planning button
def callback_step_planning():
    st.session_state.step_planning_clicked = True
    st.session_state.button_clicked_data = False
    st.session_state.button_clicked = False
    st.session_state.button_set_clicked = False
    st.session_state.button_launch_clicked = True
    st.session_state.button_info_clicked = False
    st.session_state.button_oil_sat_clicked = False
    st.session_state.button_prod_potential = False
    st.session_state.button_wd_clicked = False


# session state for info section
def session_info():
    st.session_state.button_clicked_data = False
    st.session_state.button_clicked = False
    st.session_state.button_set_clicked = False
    st.session_state.button_launch_clicked = False
    st.session_state.button_info_clicked = True
    st.session_state.button_oil_sat_clicked = False
    st.session_state.button_prod_potential = False
    st.session_state.button_wd_clicked = False


# session state for oil saturation calculation
def callback_calc():
    st.session_state.button_clicked_data = False
    st.session_state.button_clicked = False
    st.session_state.button_set_clicked = False
    st.session_state.button_launch_clicked = False
    st.session_state.button_info_clicked = False
    st.session_state.button_oil_sat_clicked = True
    st.session_state.button_prod_potential = False
    st.session_state.button_wd_clicked = False


def callback_calc_prod_p():
    st.session_state.button_clicked_data = False
    st.session_state.button_clicked = False
    st.session_state.button_set_clicked = False
    st.session_state.button_launch_clicked = False
    st.session_state.button_info_clicked = False
    st.session_state.button_oil_sat_clicked = False
    st.session_state.button_prod_potential = True
    st.session_state.button_wd_clicked = False


# session state for clay cube calculation
def callback_calc_clay():
    st.session_state.button_clicked_data = False
    st.session_state.button_clicked = False
    st.session_state.button_set_clicked = False
    st.session_state.button_launch_clicked = False
    st.session_state.button_info_clicked = False
    st.session_state.button_oil_sat_clicked = False
    st.session_state.button_prod_potential = False
    st.session_state.button_wd_clicked = False

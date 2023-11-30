from ..wellDesign import *
import streamlit as st



def callback_wd():
    st.session_state.button_clicked_data = False
    st.session_state.button_clicked = False
    st.session_state.button_set_clicked = False
    st.session_state.button_launch_clicked = False
    st.session_state.button_info_clicked = False
    st.session_state.button_oil_sat_clicked = False
    st.session_state.button_prod_potential = False
    st.session_state.button_wd_clicked = True



config = {
        'Conductor': {
            'inD': 6.25,
            'outD': 8.0,
            'weight': 58,
            'top': 0.0,
            'down': 250.0,
        },
        'Surface': {
            'inD': 6.25,
            'outD': 6.75,
            'weight': 47,
            'top': 0.0,
            'down': 3750.0,
        },
        'Production': {
            'inD': 4.75,
            'outD': 5.00,
            'weight': 39,
            'top': 0.0,
            'down': 3750.0,
        },
        'Liner': {
            'inD': 3.75,
            'outD': 4.00,
            'weight': 27,
            'top': 4800.0,
            'down': 6500.0,

               },
        'Intermediate': {
            'inD': 5.5,
            'outD': 5.75,
            'weight':39,
            'top': 0.0,
            'down': 5200.0,

                     }

        }

def design_choosing(sec, config):
    cols1, cols2, cols3, cols4, cols5 = st.columns(5)
    params = {}
 #   if config != None:

    with cols1:
        params['inD'] = st.number_input(
                    f"{sec} inner diameter, in",
                    min_value=1.0,
                    max_value=12.0,
                    value=config[sec]['inD'] if config != None else 5.0,
                    step=0.01, 
                    on_change= callback_wd
                )
    with cols2:
        params['outD'] = st.number_input(
                     f"{sec} outer diameter, in",
                    min_value=5.0,
                    max_value=20.0,
                    value=config[sec]['outD'] if config != None else 5.0,
                    step=0.01,
                    on_change= callback_wd
                )
    with cols3:  
        params['weight'] = st.number_input(
                     f"{sec} Weight, kg/in",
                    min_value=10,
                    max_value=110,
                    value=config[sec]['weight'] if config != None else 48,
                    step=1,
                    on_change= callback_wd
                )
    with cols4:
        params['top'] =  st.number_input(
                    f"{sec} top position, in",
                    min_value=0.00,
                    max_value=5000.00,
                    value=config[sec]['top'] if config != None else 0.00,
                    step=0.01,
                    on_change= callback_wd
                    )

    with cols5:
        params['down'] =  st.number_input(
                    f"{sec} low position, in",
                    min_value=0.0,
                    max_value=5000.0,
                    value=config[sec]['down'] if config != None else 0.00,
                    step=0.01,
                    on_change= callback_wd
                    )
    return params



def well_description(session_state):
    st.sidebar.subheader("Design well")

    st.sidebar.button("Design", on_click=callback_wd)
   
    if session_state.button_wd_clicked:
        cols1, cols2 = st.columns(2)
        with cols1:
            use_config = st.checkbox('Use config', ['True'])
        with cols2:
            prod_depth = st.number_input(
                                "Productive interval, in",
                                min_value=2000.00,
                                max_value=6000.00,
                                value=3000.00,
                                step=0.01, )
        wd = st.multiselect('Choose well design', ['Conductor', 'Surface', 'Intermediate', 'Production', 'Liner'], on_change=callback_wd)

        params = {}
        if wd is not None:
            for sec in wd:
                if use_config:
                    params[sec] = design_choosing(sec, config = config)
                else:
                    params[sec] = design_choosing(sec, config = None)
        

        well0 = well(name = "Test Well 001", kop = prod_depth)
        for sec in params.keys():
            t = Tubular(name = sec, inD = params[sec]['inD'], outD = params[sec]['outD'], weight = params[sec]['weight'],
                            top = params[sec]['top'], low = params[sec]['down'])
          
            well0.addTubular(t)
        
        

        plt.Figure(figsize=(10,20))

        # for w in well:
        #     
        #     well0.addTubular(w)

        # well0 = well(name = "Test Well 001", kop = 3000)
        # t0 = Tubular(name = "conductor", inD = 7.25, outD = 8, weight = 58, top = 0, low = 250)
        # t1 = Tubular(name = "surface", inD = 6.25, outD = 6.75, weight = 47, top = 0, low = 2000, shoeSize = 7)
        # t2 = Tubular(name = "intermediate", inD = 5.5, outD = 5.75, weight = 39, top = 0, low = 3750, shoeSize = 6, info = "this is very expensive!")
        # t3 = Tubular(name = "production", inD = 4.75, outD = 5, weight = 39, top = 0, low = 5200, shoeSize = 5.25)
        # t4 = Tubular(name = "liner", inD = 3.75, outD = 4, weight = 27, top = 4800, low = 6500, shoeSize = 4.5)
        # c0 = Cement(top = 0, low = 2000, tub0 = t0, tub1 = t1)
        # c1 = Cement(top = 1800, low = 3750, tub0 = t2, tub1 = t1)
        # c2 = Cement(top = 3500, low = 5200, tub0 = t2, tub1 = t3)

        
        # well0.addTubular(t0)
        # well0.addTubular(t1)
        # well0.addTubular(t2)
        # well0.addTubular(t3)
        # well0.addTubular(t4)
        # well0.addCement(c0)
        # well0.addCement(c1)
        # well0.addCement(c2)
        vis = st.button('Visualize', on_click=callback_wd)
        if vis:
            fig = well0.visualize()
            st.pyplot(fig)
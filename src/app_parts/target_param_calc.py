import sys

import numpy as np
import seaborn as sns
import streamlit as st
from src.states.states import callback_calc, callback_calc_prod_p
from src.unittests import file_path_assert
from src.utils import calculate_oil_saturation, convert_param_to_small
from src.visualization import oil_sat_plot, prod_map_plot

sys.path.append("./streamlit_app")


def archie_oil_sat(session_state):
    st.markdown("Archie equation")
    with st.expander("Description"):
        st.markdown("Archie equation is aimed to calculate water saturation in non conductive matrix rocks. "
                    "It usually works well with clean clastic sandstones and carbonate rocks")
        st.latex(r" S_w = \sqrt[n]{\frac{F R_w}{\phi^mR_t}}")
        st.markdown(" ")
        st.markdown("Where:") 
        st.markdown("$ S_{w} $ - water saturation, [frac];")
        st.markdown("$n $ - saturation exponent;")
        st.markdown("$F $ - formation factor;")
        st.markdown("$\phi$ -porosity, [frac]; ")
        st.markdown("$m $ - cementation exponent;")
        st.markdown(
            "$R_t$ - true resistivity of the formation, corrected for invasion, borehole, thin bed, and other effects,"
            " [Ohm$\cdot $ m];")
        st.markdown("$R_w$ - formation water resistivity at formation temperature, [Ohm$\cdot $ m].")


def samandoux_oil_sat(session_state):
    st.write("Simandoux equation")
    with st.expander("Description"):
        st.markdown("Simandoux equation is apllicable for shaly sandstones")
        st.latex(
            r" S_w = \sqrt{\frac{\phi^m \cdot F \cdot R_w}{\frac{(R_{sh} \cdot V_{sh})^2}{2} + F \cdot R_w \cdot \frac{R_t}{4 \cdot \phi^m} - \frac{R_{sh} \cdot V_{sh}}{2}}}"

        )
        st.markdown(" ")
        st.markdown("Where:")
        st.markdown("$ S_{w} $ - water saturation, [frac];")
        st.markdown("$R_{w}$ - formation water resistivity at formation temperature," " [Ohm$\cdot $ m];")
        st.markdown("$R_{sh}$ - shale resitivity, [Ohm$\cdot $ m];")
        st.markdown("$V_{sh}$ - shale volume, [frac];")
        st.markdown(
            "$R_t$ - true resistivity of the formation, corrected for invasion, borehole, thin bed, and other effects,"
            " [Ohm$\cdot $ m];")
        st.markdown("$\phi$ - porosity, [frac]; ")
        st.markdown("$F $ - formation factor;")
        st.markdown("$m$ - cementation exponent.")


def indonesia_oil_sat(session_state):
    st.write("Indonesia equation")
    with st.expander("Description"):
        st.markdown("The Indonesia equation may work well with fresh formation water in shaly sandstones")
        st.latex(
            r" S_w = \left[ \frac{\sqrt{\frac{1}{R_t}}}{\frac{V_{sh}^{(1-0.5V_{sh})}}{\sqrt{R_{sh}}} + \sqrt{\frac{\phi_e^m}{a \cdot R_w}}} \right]^{(2/n)}"
        )
        st.markdown(" ")
        st.markdown("Where:")
        st.markdown("$S_{w}$ - water saturation, [frac];")
        st.markdown("$R_{w}$ - formation water resistivity at formation temperature," " [Ohm$\cdot $ m];")
        st.markdown("$R_{sh}$ - shale resitivity, [Ohm$\cdot $ m];")
        st.markdown("$V_{sh}$ - shale volume, [frac];")
        st.markdown(
            "$R_t$ - true resistivity of the formation, corrected for invasion, borehole, thin bed, and other effects,"
            " [Ohm$\cdot $ m];")
        st.markdown("$\phi$ -porosity, [frac]; ")
        st.markdown("$F $ - formation factor;")
        st.markdown("$m$ - cementation exponent.")


def fertl_oil_sat(session_state):
    st.write("Fertl equation")
    with st.expander("Description"):
        st.markdown("The Fertl equation is applicable for shaly sandstones and does not depend upon $ R_{shale} $ ")
        st.latex(
            r" S_w = \phi_e^{-\left(\frac{m}{2}\right)} \left[ \sqrt{ \frac{a \cdot R_w}{R_t} + \left( \frac{\alpha \cdot V_{sh}}{2} \right)^2} - \frac{\alpha \cdot V_{sh}}{2} \right]"

        )
        st.markdown(" ")
        st.markdown("Where:")
        st.markdown("$ S_{w} $ - water saturation, [frac];")
        st.markdown("$R_{w}$ - formation water resistivity at formation temperature," " [Ohm$\cdot $ m];")
        st.markdown("$V_{sh}$ - shale volume, [frac];")
        st.markdown(
            "$R_t$ - true resistivity of the formation, corrected for invasion, borehole, thin bed, and other effects,"
            " [Ohm$\cdot $ m];")
        st.markdown("$\phi$ - porosity, [frac]; ")
        st.markdown("$ \\alpha $ - empirical constant, $ 0.25<\\alpha < 0.35 $;")
        st.markdown("$m$ - cementation exponent.")

def de_witte_oil_sat(session_state):
    st.write("De Witte equation")
    with st.expander("Description"):
        st.markdown("The De Witte equation is applicable for shaly sandstones")
        st.latex(
            r" S_w = \frac{R_w}{2 \cdot \phi}[-y + \sqrt{y^2 - (\frac{4}{R_w})(\frac{V_{sh}^2}{R_c}-\frac{1}{Rt})}]"
        )
        st.latex(r"y = V_{sh}[\frac{1}{R_w} + \frac{1}{R_c}]")
        st.markdown(" ")
        st.markdown("Where:")
        st.markdown("$ S_{w} $ - water saturation, [frac];")
        st.markdown("$ R_{c} $ - dispersed clay resistivity and can be approximated by $ R_c = 0.4 \cdot R_{sh}$, [Ohm$\cdot $ m]; " )
        st.markdown("$R_{sh}$ - shale resitivity, [Ohm$\cdot $ m];")
        st.markdown("$R_{w}$ - formation water resistivity at formation temperature," " [Ohm$\cdot $ m];")
        st.markdown("$V_{sh}$ - shale volume, [frac];")
        st.markdown(
            "$R_t$ - true resistivity of the formation, corrected for invasion, borehole, thin bed, and other effects,"
            " [Ohm$\cdot $ m];")
        st.markdown("$\phi$ - porosity, [frac]; ")
        st.markdown("$m$ - cementation exponent.")

def hossin_oil_sat(session_state):
    st.write("Hossin equation")
    with st.expander("Description"):
        st.markdown("The Hossin equation is applicable for shaly sandstones with high percent of shale varies from 10% to 30%")
        st.latex(
            r" S_w = \sqrt{ \frac{0.9}{\phi} \left[\frac{1}{R_t} - \frac{V_{sh}^2}{R_c}\right] R_w }")
        st.markdown(" ")
        st.markdown("Where:")
        st.markdown("$ S_{w} $ - water saturation fraction, [frac];")
        st.markdown("$ R_{c} $ -  dispersed clay resistivity and can be approximated by $ R_c = 0.4 \cdot R_{sh}$, [Ohm$\cdot $ m]; " )
        st.markdown("$R_{sh}$ - shale resitivity, [Ohm$\cdot $ m];")
        st.markdown("$R_{w}$ - formation water resistivity at formation temperature," " [Ohm$\cdot $ m];")
        st.markdown("$V_{sh}$ - shale volume, [frac];")
        st.markdown(
            "$R_t$ - true resistivity of the formation, corrected for invasion, borehole, thin bed, and other effects,"
            " [Ohm$\cdot $ m];")
        st.markdown("$\phi$ - porosity, [frac]. ")
       
def kamel_oil_sat(session_state):
    st.write("Kamel equation")
    with st.expander("Description"):
        st.markdown("The Kamel equation is applicable for shaly sandstones")
        st.latex(
            r" S_w = \frac{ V_{sh} \cdot R_t + \sqrt{ V_{sh}^2 \cdot R_t^2 + \frac{4 R_{sh}^2 R_t}{F \cdot R_w (1-V_{sh})^2}   }   "      
            r" } "
            

            r"  { \frac{ 2 R_{sh} R_{t} }{F \cdot R_w (1-V_{sh})^2} "
            r"}"
        )
      
        st.markdown(" ")
        st.markdown("Where:")
        st.markdown("$ S_{w} $ - water saturation fraction, [frac];")
        st.markdown("$R_{w}$ - formation water resistivity at formation temperature," " [Ohm$\cdot $ m];")
        st.markdown("$R_{sh}$ - shale resitivity, [Ohm$\cdot $ m];")
        st.markdown("$V_{sh}$ - shale volume, [frac];")
        st.markdown("$F$ - formation factor;")
        st.markdown(
            "$R_t$ - true resistivity of the formation, corrected for invasion, borehole, thin bed, and other effects,"
            " [Ohm$\cdot $ m];")
        st.markdown("$\phi$ - porosity, [frac]. ")


def calculations(session_state, options=None, type="one"):
    with st.form(key="oil_sat_form"):
        st.markdown("Empirical constant, a ")
        session_state["a"] = st.number_input(" ", min_value=1.0, max_value=100000.0, value=1.0, step=0.01)
        st.markdown("Resistivity of water formation, $R_{w}$ [Ohm$\cdot $ m] ")
        session_state["Rw"] = st.number_input(" ", min_value=0.08, max_value=100000.0, value=0.2, step=0.01)

        if options not in  ["Archie",'Fertl']:
            st.markdown("Shale resitivity, $R_{sh}$ [Ohm$\cdot $ m]")
            session_state["R_sh"] = st.number_input(" ", min_value=0.08, max_value=100000.0, value=0.15, step=0.01)

        # if options in ['Fertl' ,'Indonesia' ,'De Witte', 'Hossin', 'Kamel']:
        #     st.markdown("$R_{t}$ - reservoir temperature, [°C];")
        #     session_state['Rt'] = st.number_input(" ", min_value=0.08, max_value=100.0, value=0.2, step=0.01) 

        st.markdown("Saturation exponent, $n$")
        session_state["n"] = st.number_input(" ", min_value=1.8, max_value=4.0, value=2.0, step=0.01)
        st.markdown("Cementation exponent, $m$")
        session_state["m"] = st.number_input(" ", min_value=1.7, max_value=3.0, value=2.0, step=0.01)

        st.markdown("Formation factor, $F$")
        session_state["F_factor"] = st.number_input(" ", min_value=0.5, max_value=5.0, value=3.0, step=0.01)

        submit_button = st.form_submit_button(label="Calculate")
        if submit_button:
            if type == "one":
                final_oil_sat = calculate_oil_saturation(session_state, type=options)
            elif isinstance(type, list):
                oil_sat_l = []
                for i, t in enumerate(type):
                    if t == True:
                        if i == 0:
                            oil_sat_l.append(calculate_oil_saturation(session_state, type="Archie"))
                        elif i == 1:
                            oil_sat_l.append(calculate_oil_saturation(session_state, type="Samandoux"))
                        elif i == 2:
                            oil_sat_l.append(calculate_oil_saturation(session_state, type="Indonesia"))
                        elif i == 3:
                            oil_sat_l.append(calculate_oil_saturation(session_state, type="Fertl"))
                        elif i == 4:
                            oil_sat_l.append(calculate_oil_saturation(session_state, type="De Witte"))
                        elif i == 5:
                            oil_sat_l.append(calculate_oil_saturation(session_state, type="Hossin"))
                        elif i == 6:
                            oil_sat_l.append(calculate_oil_saturation(session_state, type="Kamel"))

                final_oil_sat = np.zeros_like(oil_sat_l[0])
                for val in oil_sat_l:
                    final_oil_sat = np.add(final_oil_sat, val)
                final_oil_sat = final_oil_sat / len(oil_sat_l)
            # clipping 
            final_oil_sat = np.clip(final_oil_sat, 0, np.inf)
            session_state["oil_saturation_file"] = final_oil_sat
            session_state["parameter_for_opt"] = "Oil saturation"
            st.success(f"{options} oil saturation is sucessfully calculated")
            sns.set_style("whitegrid")
            oil_sat_plot(session_state)


def prod_potential():
    st.write("Productivity potential ")
    with st.expander("Description"):
        st.latex(r"K = P \cdot \phi \cdot S_o")
        st.markdown(" ")
        st.markdown("Where $ S_o $ - oil saturation, [frac]")
        st.markdown(" $ \phi $ - porosity, [frac]")
        st.markdown("K - permeability, [md]")


def targ_parm_calc(session_state):
    st.sidebar.subheader("Target parameter calculation")
    session_state["calculation_button"] = st.sidebar.button("Calculate", on_click=callback_calc)
    if session_state.button_oil_sat_clicked:
        tab1, tab2 = st.tabs(["Oil saturation", "Productivity potential"])

        with tab1:
            #  if session_state['parameter_for_opt'] == 'Oil saturation':
            error = file_path_assert(session_state, exclude=['oil_saturation', 'prod_potential','permeability'])
        
        
            if not error:
                st.subheader("Calculation of oil saturation")
                radiobox = st.radio(
                    "Choose a method",
                    [
                        "Choose an equation",
                        "Average over formulas",
                    ],
                )

                if radiobox == "Choose an equation":
                    options = st.radio(
                        "Choose equation for oil saturation calculation",
                        ("Archie", "Samandoux", "Indonesia", "Fertl", "De Witte", "Hossin", "Kamel"),
                    )

                    if options == "Archie":
                        archie_oil_sat(session_state)
                    elif options == "Samandoux":
                        samandoux_oil_sat(session_state)
                    elif options == "Indonesia":
                        indonesia_oil_sat(session_state)
                    elif options == "Fertl":
                        fertl_oil_sat(session_state)
                    elif options == "De Witte":
                           de_witte_oil_sat(session_state)
                    elif options == "Hossin":
                           hossin_oil_sat(session_state)
                    elif options == "Kamel":
                           kamel_oil_sat(session_state)
                    calculations(session_state, options)

                elif radiobox == "Average over formulas":
                    cols = st.columns(8, gap="large")
                    # archie
                    with cols[0]:
                        st.write("Archie")
                        archie = st.checkbox(" ", key="check1", value = True)
                        #archie_oil_sat(session_state)
                    # Samandoux
                    with cols[1]:
                        st.write("Simandoux")
                        samandoux = st.checkbox(" ", key="check2", value = True)
                        
                        #samandoux_oil_sat(session_state)
                    # Indonesia
                    with cols[2]:
                        st.write("Indonesia")
                        indonesia = st.checkbox(" ", key="check3", value = True)
                        #indonesia_oil_sat(session_state)
                    # Fertl
                    with cols[3]:
                        st.write("Fertl")
                        fertl = st.checkbox(" ", key="check4")
                        #fertl_oil_sat(session_state)
                    # De Witte
                    with cols[4]:
                        st.write("De Witte")
                        de_witte = st.checkbox(" ", key="check5")
                        #de_witte_oil_sat(session_state)
                    # Hossin
                    with cols[5]:
                        st.write("Hossin")
                        hossin = st.checkbox(" ", key="check6")
                        #hossin_oil_sat(session_state)
                    # Kamel
                    with cols[6]:
                        st.write("Kamel")
                        kamel = st.checkbox(" ", key="check7")
                        #kamel_oil_sat(session_state)

                    calculations(session_state, type=[archie, samandoux, indonesia, fertl, de_witte, hossin, kamel])
        with tab2:
            # elif session_state['parameter_for_opt'] == 'Productivity potential':
            # session_state['calculation_button_prm'] = st.sidebar.button(
            #     'Calculate', on_click=callback_calc_prod_p, key='prod_map')

            error = file_path_assert(session_state, exclude=['resistivity','prod_potential'])
            #error = False
            #error = file_path_assert(session_state, exclude=['oil_saturation'])
            prod_potential()
            if not error:
                st.subheader("Calculation of productivty potential")

                calc_prod_map = st.button("Calculate productivity potential")
                if calc_prod_map:
                    session_state["prod_potential"] = (
                        session_state.permeability_file
                        * session_state.porosity_file
                        * session_state.oil_saturation_file
                    )
                    # clip productivity potential
                    session_state["prod_potential"] = np.clip(session_state["prod_potential"], 0, np.inf)
                    session_state["parameter_for_opt"] = "Productivity potential"
                    prod_map_plot(st.session_state)
                    st.success(f"Productivity potential is sucessfully calculated")
                    sns.set_style("whitegrid")
            
                

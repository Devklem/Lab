import sys

import streamlit as st

sys.path.append("./streamlit_app")
from src.states import session_info


def guide_section():
    st.sidebar.subheader("Guidance")

    set_button = st.sidebar.button("Guide", on_click=session_info)

    if st.session_state.button_info_clicked:
        page1, page2, page3 = st.tabs(
            [
                "Main idea behind application",
                "Guide",
                "Differential evolution algorithm description",
            ]
        )
        with page1:
            st.markdown(
                """
            <style>
            [data-testid="stMarkdownContainer"] ul{
                list-style-position: inside;
            }
            </style>
            """,
                unsafe_allow_html=True,
            )
            st.write(
                "Algorithm is based on optmizing objective function which is based on resistivity data and "
                "angle constraints which should be followed during well drilling. Objective function is the following:"
            )
            st.latex(
                r"""
             \large g_f(N(x_i)) = \frac{\sum_{i=0}^{n}  \int_{x_i}^{x_{i+n_s}} f(x)dx}{\sum_{i=0}^{n}||x_{i+{n_s} - x_i } ||}
            """
            )
            st.markdown("Well coordinates are calculated by changing zenith and azimuth angles")
            st.latex(
                r"\large x_{i+1} = x_{i} +l \cdot sin(\theta_i)cos(\phi_i) \\"
                r"\large y_{i+1} = y_{i} +l \cdot sin(\theta_i)sin(\phi_i) \\"
                r"\large z_{i+1} = z_{i} +l \cdot sin(\theta_i)"
            )
            st.markdown(r"where: $ \theta $ - zenith angle, $\phi $ - azimuth")
            st.markdown(r"Dogleg constraint")
            st.latex(
                r"\large DLS = {cos^{-1}}[(cos \theta_1 \cdot cos \theta_2) + (sin \theta_1 \cdot sin \theta_2) \cdot cos(\phi_2 - \phi_1)]\cdot \frac{100}{CL}"
            )
        with page2:
            st.markdown(
                "### Step by step guide how to work with application",
                unsafe_allow_html=True,
            )
            st.markdown("- Step 1:")
            st.markdown("Upload your data. ")
            st.markdown(
                "For the first time you need to upload file. Once it is uploaded you can select file from the existing and work with him remotedly "
            )
            st.markdown("- Step 2:")
            st.markdown("**Set the parameters of the well**")
            st.markdown(
                "**Initial position:** cartesian coordinates of the first point of the well."
                "Drilling direction oriented along the Z axis"
            )
            st.markdown("**Initial zenith and azimuth angles:** null points of angles ")
            st.markdown("**Azimuth angle constraint:** maximum azimuth allowed. Minimum value is 0")
            st.markdown(
                "**Zenith angle constraint:** maximum zenith allowed. As we work in productive reservoir and"
                "well desing in this region is horizontal, there is a requirement do not use zenith angle more than "
                "92 degrees"
            )
            st.markdown("**Dogleg constraint:** main angle constraint used for algorithm optimization ")

            st.markdown("- Step 3:")
            st.markdown("Optionally select the parameters of the choosen algorithm for geosteering")

            st.markdown("- Step 4:")
            st.markdown("Go to simulation page and choose the prefferable option for modeling")
            st.markdown("Full planning button. Entire trajectory will be constructed")
            st.markdown(
                'Step planning button. Trajectory will be built on selected steps. In order to continiue building the trajectory by steps click on "Step planning" again'
            )

        with page3:
            st.markdown(
                "The differential evolution method is stochastic in nature. It does not use gradient methods to find the minimum, and can search large areas of candidate space, but often requires"
                " larger numbers of function evaluations than conventional gradient-based techniques."
            )
            st.markdown("Differential evolution has the following parameters to vary:")
            st.markdown("- **Strategy**. The differential evolution strategy to use")
            st.markdown(
                "- **Maximum generations (maxiter)**.  The maximum number of generations over which the entire population is evolved. "
                "The maximum number of function evaluations (with no polishing) is: $(maxiter + 1)$"
            )
            st.markdown(
                "- **Population size (popsize)**. A multiplier for setting the total population size. The population has $ popsize * N $individuals. This keyword is overridden if an initial population is supplied via the init keyword. "
                'When using init="sobol" the population size is calculated as the next power of 2 after $ popsize * N. $ '
            )

            st.markdown(
                "- **Mutation**. The mutation constant. In the literature this is also known as differential weight, being denoted by F. "
                "If specified as a float it should be in the range [0, 2]. If specified as a tuple (min, max) dithering is employed. "
                "Dithering randomly changes the mutation constant on a generation by generation basis. The mutation constant for that generation is taken from U[min, max). "
                "Dithering can help speed convergence significantly."
                " Increasing the mutation constant increases the search radius, but will slow down convergence."
            )
            st.markdown(
                "- **Recombination**. The recombination constant, should be in the range [0, 1]. In the literature this is also known as the crossover probability,"
                " being denoted by CR. Increasing this value allows a larger number of mutants to progress into the next generation, but at the risk of population stability."
            )

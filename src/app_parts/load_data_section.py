import pickle
from pathlib import Path
from src.unittests import check_types, check_well_types, check_loads
from src.states import callback_data
from src.utils import save_uploadedfile, upload_selected_file, file_selector
import sys

import streamlit as st

sys.path.append("./streamlit_app")

folder_path_res = str(Path("data/raw/res"))
folder_path_por = str(Path("data/raw/por"))
folder_path_traj = str(Path("data/raw/well"))
folder_path_clay = str(Path("data/raw/clay"))
folder_path_perm = str(Path("data/raw/perm"))





def load_data_section(session_state):
    st.sidebar.subheader("Input data")

    session_state["data_button"] = st.sidebar.button("Load data", on_click=callback_data)
    if session_state.button_clicked_data:
        st.markdown("### Upload planned well trajectory")
        session_state["type_of_load_traj"] = st.radio(
            " ",
            ["Select from existing file", "Upload file"],
            on_change=callback_data,
            key="traj",
        )
        if session_state["type_of_load_traj"] == "Upload file":
            uploaded_file = st.file_uploader(
                "Specify the path to planned well trajectory:",
                on_change=callback_data,
                key="traj2",
            )
            if uploaded_file is not None:
                file = pickle.load(uploaded_file)
                save_uploadedfile(uploaded_file, type="trajectory")

                session_state["planned_traj"] = check_well_types(file)

        elif session_state["type_of_load_traj"] == "Select from existing file":
            path = file_selector(folder_path_traj, on_change=callback_data)
            session_state["traj_path"] = path
            submit_upload = st.button("Submit upload", key="sub2", on_click=callback_data)
            
            if submit_upload:
                file = upload_selected_file(session_state["traj_path"])
                st.success("Planned trajectory {} is successfully loaded".format((path.split("/")[-1])), icon="✅")
                session_state["planned_traj"] = check_well_types(file)
          

        st.subheader("Upload volume data")
        tabs1, tabs2, tabs3, tabs4 = st.columns(4)

        with tabs1:
            st.markdown("##### Clay cube")
            session_state["type_of_load_clay"] = st.radio(
                " ",
                ["Select from existing file", "Upload file"],
                on_change=callback_data,
                key="clay",
            )
            if session_state["type_of_load_clay"] == "Upload file":
                uploaded_file = st.file_uploader(
                    "Specify the path to planned well trajectory:",
                    on_change=callback_data,
                    key="clay2",
                )
                if uploaded_file is not None:
                    file = pickle.load(uploaded_file)
                    save_uploadedfile(uploaded_file, type="clay")
                
                    session_state["clay_cube"] = check_types(file, session_state)

            elif session_state["type_of_load_clay"] == "Select from existing file":
                path = file_selector(folder_path_clay, on_change=callback_data)
                session_state["file_path"] = path
                
                st.session_state.submit_clay = st.button("Submit upload", key="clay_sub")
                if st.session_state.submit_clay:
                    file = upload_selected_file(session_state["file_path"])
                    st.success("Dataset {} is successfully loaded".format((path.split("/")[-1])), icon="✅")
                    (
                        session_state["clay_cube"],
                        session_state["x_cube"],
                        session_state["y_cube"],
                        session_state["z_cube"],
                    ) = check_types(file, session_state)

        with tabs2:
            st.markdown("##### Resistivity")
            session_state["type_of_load_res"] = st.radio(
                " ",
                ["Select from existing file", "Upload file"],
                on_change=callback_data,
                key="res",
            )
            if session_state["type_of_load_res"] == "Upload file":
                uploaded_file = st.file_uploader(
                    "Specify the path to the resistivity data:",
                    on_change=callback_data,
                    key="resf",
                )
                if uploaded_file is not None:
                    file = pickle.load(uploaded_file)
                    save_uploadedfile(uploaded_file, type="resistivity")

                    (
                        session_state.file,
                        session_state["x_cube"],
                        session_state["y_cube"],
                        session_state["z_cube"],
                    ) = check_types(file, session_state)

            elif session_state["type_of_load_res"] == "Select from existing file":
                path = file_selector(folder_path_res, on_change=callback_data)
                session_state["file_path"] = path
                submit_upload = st.button("Submit upload", key="sub1")
                if submit_upload:
                    file = upload_selected_file(session_state["file_path"])
                    st.success("Dataset {} is successfully loaded".format((path.split("/")[-1])), icon="✅")
                    (
                        session_state.file,
                        session_state["x_cube"],
                        session_state["y_cube"],
                        session_state["z_cube"],
                    ) = check_types(file, session_state)

        with tabs3:
            st.markdown("##### Porosity")
            session_state["type_of_load_por"] = st.radio(
                " ",
                ["Select from existing file", "Upload file"],
                on_change=callback_data,
                key="por",
            )
            if session_state["type_of_load_por"] == "Upload file":
                uploaded_file = st.file_uploader(
                    "Specify the path to  porosity data:",
                    on_change=callback_data,
                    key="porf",
                )
                if uploaded_file is not None:
                    porosity_file = pickle.load(uploaded_file)
                    save_uploadedfile(uploaded_file, type="porosity")

                    (
                        session_state.porosity_file,
                        session_state["x_cube"],
                        session_state["y_cube"],
                        session_state["z_cube"],
                    ) = check_types(porosity_file, session_state)
            elif session_state["type_of_load_por"] == "Select from existing file":
                path = file_selector(folder_path_por, on_change=callback_data)
                session_state["porosity_file_path"] = path
                submit_upload_por = st.button("Submit upload", key="sub3")
                if submit_upload_por:
                    porosity_file = upload_selected_file(session_state["porosity_file_path"])
                    st.success("Dataset {} is successfully loaded".format((path.split("/")[-1])), icon="✅")
                    (
                        session_state.porosity_file,
                        session_state["x_cube"],
                        session_state["y_cube"],
                        session_state["z_cube"],
                    ) = check_types(porosity_file, session_state)

        with tabs4:
            st.markdown("##### Permeability")
            session_state["type_of_load_perm"] = st.radio(
                " ",
                ["Select from existing file", "Upload file"],
                on_change=callback_data,
                key="perm",
            )
            if session_state["type_of_load_perm"] == "Upload file":
                uploaded_file = st.file_uploader(
                    "Specify the path to permeability data:",
                    on_change=callback_data,
                    key="permf",
                )
                if uploaded_file is not None:
                    permeability_file = pickle.load(uploaded_file)
                    save_uploadedfile(uploaded_file, type="permeability")

                    (
                        session_state.permeability_file,
                        session_state["x_cube"],
                        session_state["y_cube"],
                        session_state["z_cube"],
                    ) = check_types(permeability_file, session_state)
            elif session_state["type_of_load_perm"] == "Select from existing file":
                path = file_selector(folder_path_perm, on_change=callback_data)
                session_state["perm_file_path"] = path
                submit_upload_por = st.button("Submit upload", key="sub4")
                if submit_upload_por:
                    permeability_file = upload_selected_file(session_state["perm_file_path"])
                    st.success("Dataset {} is successfully loaded".format((path.split("/")[-1])), icon="✅")
                    (
                        session_state.permeability_file,
                        session_state["x_cube"],
                        session_state["y_cube"],
                        session_state["z_cube"],
                    ) = check_types(permeability_file, session_state)

    check_loads(session_state)
   

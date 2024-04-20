import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    st.write("### Quick Project Summary")

    st.info(
        f"**General Information**\n\n"
        f"Powdery mildew is a common fungal disease affecting cherry trees, caused by *Podosphaera clandestina*."
        f" Infected leaves develop a layer of white powdery growth on their surfaces, leading to reduced photosynthesis and crop yield."
        f" This project aims to differentiate between healthy and infected leaves using visual analysis techniques."
        f"\n\n"
        f"Visual criteria for detecting infected leaves include:\n\n"
        f"* Light-green circular lesions on leaf surfaces, evolving into a subtle white cotton-like growth."
        f"\n\n")

    st.warning(
        f"**Project Dataset**\n\n"
        f"The dataset comprises 2104 images of healthy cherry leaves and 2104 images of leaves affected by powdery mildew."
        f" Each leaf was individually photographed against a neutral background."
        )

    st.success(
        f"The project fulfills three key business requirements:\n\n"
        f"1. Conducting a study to visually differentiate healthy and infected leaves.\n\n"
        f"2. Developing a model capable of accurately predicting whether a given leaf is infected by powdery mildew.\n\n"
        f"3. Providing a downloadable prediction report for the examined leaves."
        )

    st.write(
        f"For further details, please consult the "
        f"[Project README file](https://github.com/EMPZsolt/my-fifth-project#readme).")
import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Code adapted from the Churnometer walkthrough project by Code Institute.
# Create dataframe with house data
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_house_data():
    df = pd.read_csv("outputs/datasets/collection/HousePricesRecords.csv")
    return df


# Load PKL file
def load_pkl_file(file_path):
    return joblib.load(filename=file_path)

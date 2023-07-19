import streamlit as st
from src.data_management import load_house_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ppscore as pps
sns.set_style("whitegrid")


def page2_house_price_study():

    # load data
    df = load_house_data()

    # hard copied from sales price correlation study notebook
    vars_to_study = ['1stFlrSF', 'GarageArea', 'GrLivArea',
                     'KitchenQual', 'TotalBsmtSF', 'OverallQual', 'YearBuilt']

    st.write("## House Price Correlation Study")

    st.info(
        f"** Business requirement 1** \n\n"
        f"* The client is interested in discovering how the house "
        f" attributes correlate with the sale price."
    )

    # inspect data
    if st.checkbox("Inspect House Dataset"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"Find below the first 10 rows of the dataset.")

        st.write(df.head(10))

    st.write("---")

    st.write(
        f"We conducted a correlation study in the notebook to better "
        f" understand how the variables are correlated to the sales price.\n "
        f"We found the following most correlated "
        f" variables: **{vars_to_study}**"
    )

    # Text based on "SalePrice Correlation" notebook - "Conclusions" section
    st.info(
        f"**From the Scatterplot, coupled with the Correlation "
        f"and PPS analysis, we were able to draw the following insight:**\n "
        f"* There is mostly a postive correlation with houses of "
        f"higher overall quality material and finish of the "
        f" house with sales price.\n "
        f"* Large first floors, garages,above-ground living areas "
        f"and basements in square feet tend to sell for higher price.\n"
        f"* Rececently built houses tend to sell at higher prices.\n\n "
    )

    st.info(
        f"* Our client expects a data visualisations of the correlated "
        f" variables against the sale price."
    )

    st.write(
        f"We used Spearman and Pearson Correlation and the "
        f"Power Predictive Score (PPS) Heatmap to demonstrate  "
        f"variable correlations below"
    )

# Code copied from 'SalePrice study' notebook - Correlation Study section

    # Checkbox widget displays the Spearman correlation Heatmap
    if st.checkbox("Spearman Correlations"):
        st.write(
            f"* This plot displays a monotonic correlation and a threshold "
            f" set to 0.6 (moderate correlation).\n"
        )
        df_corr_pearson, df_corr_spearman, pps_matrix = CalculateCorrAndPPS(df)
        heatmap_corr(df=df_corr_spearman, threshold=0.6,
                     figsize=(20, 12), font_annot=12)

    # Checkbox widget displays the Pearson correlation Heatmap
    if st.checkbox("Peason Correlations"):
        st.write(
            f"* This plot displays a linear correlation and a threshold "
            f" set to 0.6 (moderate correlation).\n"
        )
        df_corr_pearson, df_corr_spearman, pps_matrix = CalculateCorrAndPPS(df)
        heatmap_corr(df=df_corr_pearson, threshold=0.6,
                     figsize=(20, 12), font_annot=15)

    # Checkbox widget displays the PPS Heatmap
    if st.checkbox("Predictive Power Score"):
        st.write(
            f"* This plot displays correlation btween variable x and y "
            f"and a threshold set to 0.15 (moderate correlation).\n"
        )
        df_corr_pearson, df_corr_spearman, pps_matrix = CalculateCorrAndPPS(df)
        heatmap_pps(df=pps_matrix, threshold=0.15,
                    figsize=(20, 12), font_annot=15)

    st.write("---")

    st.write(
        f"The scatterplots displays each selected variable "
        f" against the SalePrice. "
    )

    # Create a Dataframe for the Selected Variables
    # for study with the SalesPrice
    df_to_study = df.filter(vars_to_study + ['SalePrice'])

    if st.checkbox("Variable Correlation to Sale Price"):
        Var_corr_to_sale_price(df_to_study)


# Displays heatmaps for correlation and PPS
def heatmap_corr(df, threshold, figsize=(20, 12), font_annot=15):
    if len(df.columns) > 1:
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        mask[abs(df) < threshold] = True

        fig, axes = plt.subplots(figsize=figsize)
        sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True,
                    mask=mask, cmap='viridis',
                    annot_kws={"size": font_annot}, ax=axes,
                    linewidth=0.5
                    )
        axes.set_yticklabels(df.columns, rotation=0, fontsize=20)
        axes.set_xticklabels(df.columns, fontsize=20)
        plt.ylim(len(df.columns), 0)
        st.pyplot(fig)


def heatmap_pps(df, threshold, figsize=(20, 12), font_annot=15):
    if len(df.columns) > 1:
        mask = np.zeros_like(df, dtype=np.bool)
        mask[abs(df) < threshold] = True
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True,
                         mask=mask, cmap='rocket_r',
                         annot_kws={"size": font_annot},
                         linewidth=0.05, linecolor='grey')
        ax.set_yticklabels(df.columns, rotation=0, fontsize=20)
        ax.set_xticklabels(df.columns, fontsize=20)
        plt.ylim(len(df.columns), 0)
        st.pyplot(fig)


# Function to calculate correlations and PPS
def CalculateCorrAndPPS(df):
    df_corr_spearman = df.corr(method="spearman")
    df_corr_pearson = df.corr(method="pearson")

    pps_matrix_raw = pps.matrix(df)
    pps_matrix = pps_matrix_raw.filter(['x', 'y', 'ppscore']).pivot(
        columns='x', index='y', values='ppscore')

    return df_corr_pearson, df_corr_spearman, pps_matrix


# Functions to display scatterplots to show correlations
def Var_corr_to_sale_price(df_to_study):
    target_var = 'SalePrice'
    for col in df_to_study.drop([target_var], axis=1).columns.to_list():
        plot_scatter(df_to_study, col, target_var)
        st.write("\n\n")


def plot_scatter(df, col, target_var):
    fig, axes = plt.subplots(figsize=(8, 4))
    sns.scatterplot(data=df, x=col, y=target_var)
    plt.title(f"{col}", fontsize=20, y=1.05)
    st.pyplot(fig)

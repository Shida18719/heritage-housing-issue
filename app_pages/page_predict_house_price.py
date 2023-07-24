import streamlit as st
import pandas as pd
from datetime import date
from src.data_management import load_house_data, load_pkl_file
from src.machine_learning.predictive_analysis_ui import predict_house_price


def page4_predict_house_price():

    # load required files for predicting house price
    version = 'v2'
    regressor_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_saleprice/{version}/best_regressor_pipeline.pkl")
    house_features = (pd.read_csv(
        f"outputs/ml_pipeline/predict_saleprice/{version}/X_train.csv"
    ).columns.to_list())

    st.write("### Predict House Price")

    st.info(
        f"### Business requirement 2 \n\n"
        f"The client is interested in predicting the house sale price from her"
        f" four inherited houses and any other house in Ames, Iowa. \n\n"
        f"* The table below displays the profile of the "
        f" four inherited houses.\n "
        f" Slide left and right to view the house attributes.\n\n"
    )

    # load inherited houses file

    inherited_df = pd.read_csv(
        "inputs/datasets/raw/house-price-20211124T154130Z-001/house-price/inherited_houses.csv")

    st.write(inherited_df.head())

    st.write("---")

    st.success(
        f" The table below displays the 5 most relevant features "
        f" used by the ML model for predicting house sale price. \n"
        f" These features will be required by our client in order to maximize "
        f" sale price on her inherited houses \n\n"
    )

    inherited_df = inherited_df.filter(house_features)

    st.write(inherited_df.head())

    st.write("Below are estimated individual value predicted for the 4"
             " inherited houses.\n\n "
             "* $129,057.7336 \n "
             "* $153,333.30 \n "
             "* $154,700.70\n "
             "* $182,183.90\n "
             )

    st.write("---")

    st.write("### Predict House Prices in Ames, Iowa  \n")

    st.info(
        f"The ML model was trained on the 5 selected features and it "
        f" will be use to predict house prices in Ames, Iowa.\n\n"
        f"* OverallQual: Rates the overall material and finish of the house"
        f" Rating from 1 (Very Poor) to 10 (Very Excellent)\n "
        f"* GrLiveArea: Above grade (ground) living area square "
        f" feet(max 11,284 sq feet)\n "
        f"* TotalBsmtSF: Total square feet of basement "
        f" area (max of 12,220 sq feet)\n "
        f"* GarageArea: Size of garage in square feet(max of 2,836 sq feet)\n "
        f"* GarageYrBlt: Year garage was built (1900 to current year)\n"
    )

    st.success(
        f" OverallQual and GrLiveArea has the most significant relative "
        f" correlation with the house sale price prediction.\n "
        f"* With a Pearson of 0.79 and Spearman score of 0.8, while a PPS "
        f" of 0.4 score for OverallQual.\n "
        f"* Spearman correlation 0f 0.73 and Pearson score of 0.71 for "
        f" GrLiveArea, both indicate a strong correlation with sales price.\n"
    )
    X_live = DrawInputsWidgets()

    if st.button("Run Predictive Analysis"):
        price_prediction = predict_house_price(
            X_live, house_features, regressor_pipe)

        # Create a logic to display the results and currency value in Dollar
        price = price_prediction
        value = float(price.round(1))

        amount = '${:,.2f}'.format(value)

        statement = (
            f"* Based on the input values, we estimate this house "
            f" to be worth **{amount}**"
        )

        # Return price prediction value for house price
        st.write(statement)


# Create input widgets to feed the data values to df for predictions
def DrawInputsWidgets():

    # load dataset
    df = load_house_data()
    percentageMin, percentageMax = 0.4, 2.0

    # Create input widgets for the five best features in two rows
    col1, col2 = st.beta_columns(2)
    col3, col4, col5 = st.beta_columns(3)

    # Create an empty DataFrame, for the live data
    X_live = pd.DataFrame([], index=[0])

    # draw the widget based on the variable type
    # and set initial values
    with col1:
        feature = "OverallQual"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].sort_values(ascending=True).unique()
        )

    X_live[feature] = st_widget

    with col2:
        feature = "GrLivArea"
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min()*percentageMin),
            max_value=int(df[feature].max()*percentageMax),
            value=int(df[feature].median()),
        )

    X_live[feature] = st_widget

    with col3:
        feature = "TotalBsmtSF"
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min()*percentageMin),
            max_value=int(df[feature].max()*percentageMax),
            value=int(df[feature].median()),
        )
    X_live[feature] = st_widget

    with col4:
        feature = "GarageArea"
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min()*percentageMin),
            max_value=int(df[feature].max()*percentageMax),
            value=int(df[feature].median()),)

    X_live[feature] = st_widget

    with col5:
        feature = "GarageYrBlt"
        st_widget = st.number_input(
            label=feature,
            min_value=1900,
            max_value=date.today().year,
            value=int(df[feature].median()),
        )
    X_live[feature] = st_widget

    return X_live

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_house_data, load_pkl_file
from src.machine_learning.evaluate_regression import (
    regression_performance, regression_evaluation, regression_evaluation_plots)


def page5_modelling_and_evaluation():

    # load pipeline files
    version = 'v2'
    house_price_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_saleprice/{version}/best_regressor_pipeline.pkl")
    feat_importance = plt.imread(
        f"outputs/ml_pipeline/predict_saleprice/{version}/features_importance.png")
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_saleprice/{version}/X_train.csv")
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_saleprice/{version}/X_test.csv")
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_saleprice/{version}/y_train.csv")
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_saleprice/{version}/y_test.csv")

    st.write("### ML Pipeline Performance")

    # Summary of the ML Pipeline conclusion
    st.info(
        f"We address the second business requirement using a regressor model"
        f" to predict the sales price of houses in Ames, Iowa.\n"
        f"* We knew that a **Regressor Model** would be suitable for "
        f" predicting the **Sale Price** of the houses since our **target** "
        f" is a continous variable.\n "
        f"* We have completed a series of steps used for **Supervised  "
        f" Learning** and following the **CRISP-DM** workflow.\n "
        f"* We have carefully considered the following steps after splitting "
        f" the data into Train and Test set:\n "
        f"* The data cleaning, feature engineering, feature scaling, feature"
        f" selection and modelling steps in order to achieve the results.\n "
        f"* The regressor performance met the requirement.\n "
        f"* We first used the default parameter, then later"
        f" configured the hyperparameter for optimization and using fewer "
        f" variables to deliver equivalent results. \n"
        f"* The pipeline met the client's performance requirement.\n"
        f"* We aimed at an R2 score of at least 0.75 for both the "
        f" Train and Test sets \n\n"
        f"* We achieved an R2 score of **0.91** on the Train and **0.81** "
        f" on the Test set. Therefore, deemed good fit for the data\n\n"
    )

    st.write("---")

    # show pipeline steps
    st.write(" ML pipeline to predict house Sale Price")
    st.write(house_price_pipe)

    st.write("---")

    # show best features importance
    st.write("The features importance the model was trained on shown below.\n"
             )
    st.write(X_train.columns.to_list())
    st.image(feat_importance)

    st.write("---")

    # Evaluate pipline performance on train and test sets
    st.write("### ML Pipeline Performance")

    regression_performance(X_train, y_train, X_test, y_test, house_price_pipe)

    st.write("---")

    # Predicted versus actual sale price plot for train and test sets
    st.write("* **Predicted Price versus actual Sale Price Scatterplot**")

    st.info("**We compared the predicted price to the actual price "
            " values in the plots shown below**.\n "
            "* The plots that Predict y and x Actual plot, is"
            " positively sloped, it indicates a positive relationship between "
            " the Predict and Actual variable, they tend to follow the "
            " actual value(the red diagonal line), which "
            " indicate predicted price is equals to the actual price.\n"
            "* Where the prediction data point equals to actual value is "
            " just slightly above $400000.\n"
            "* The model may not predict prices well above $425000 accurately,"
            " as the data points spread indicates greater variability "
            " and below the red line.\n\n "
            " Note: Plot may take some time to load."
            )

    regression_evaluation_plots(
        X_train, y_train, X_test, y_test, house_price_pipe)

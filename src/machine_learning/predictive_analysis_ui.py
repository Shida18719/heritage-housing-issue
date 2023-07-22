# Import streamlit library
import streamlit as st


# Run prediction from pipeline

def predict_house_price(X_live, house_features, regressor_pipe):

    # from live data, subset features related to this pipeline
    X_live = X_live.filter(house_features)

    # Predict

    price_prediction = regressor_pipe.predict(X_live)
    return price_prediction

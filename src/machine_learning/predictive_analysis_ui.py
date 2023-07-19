# Import streamlit library
import streamlit as st


# Run prediction from pipeline

def predict_house_price(X_live, house_features, regressor_pipe):

    # from live data, subset features related to this pipeline
    X_live = X_live.filter(house_features)

    # Predict

    price_prediction_proba = regressor_pipe.predict(X_live)

    # Create a logic to display the results and currency value in Dollar

    proba = price_prediction_proba

    value = float(proba.round(1))

    amount = '${:,.2f}'.format(value)

    statement = (
        f"* Based on the input values, we estimate this house "
        f" to be worth **{amount}**"
    )

    # Return price prediction value for house price
    st.write(statement)

    st.write(proba)

# Import streamlit library
import streamlit as st

# Run prediction from pipeline
def predict_house_price(X_live, house_features, regressor_pipe):

    # from live data, subset features related to this pipeline
    X_live = X_live.filter(house_features)

    # predict
	
    price_prediction_proba = regressor_pipe.predict(X_live)

    # create a logic to display the results
	
    proba = price_prediction_proba
	
    value = float(proba.round(1))
	
    amount = '${:,.2f}'.format(value)
	
    statement = (
		f'* Based on the input values, we estimate this house to be worth **{amount}**'
	)

    # return price_prediction
    st.write(statement)

    st.write(proba)
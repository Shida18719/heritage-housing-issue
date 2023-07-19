import streamlit as st
from app_pages.multi_page import MultiPage


# load pages scripts
from app_pages.page_project_summary import page1_project_summary
from app_pages.page_house_price_study import page2_house_price_study
from app_pages.page_hypothesis import page3_hypothesis_and_validation
from app_pages.page_predict_house_price import page4_predict_house_price
from app_pages.page_modelling import page5_modelling_and_evaluation


# Create an instance of the app
app = MultiPage(app_name="Heritage Housing")

# Add app pages using .add_page()
app.add_page("Project Summary", page1_project_summary)
app.add_page("House Price Correlation Study", page2_house_price_study)
app.add_page("Hypothesis and Validation", page3_hypothesis_and_validation)
app.add_page("Predict House Price", page4_predict_house_price)
app.add_page("Modelling and Evaluation", page5_modelling_and_evaluation)


app.run()  # Run the app

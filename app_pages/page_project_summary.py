import streamlit as st


def page1_project_summary():

    st.write("### Project Summary")

    st.write(
        f" A Data Analytics and ML Web App to predict and visualise "
        f" the Sales of house price\n"
        f" from a four inherited house and any other houses within the "
        f" surrounding area of Ames.\n"
        f" The aim is to allow users to discover the correlation "
        f" between various house attributes and sale price.\n"
        f" Also, to predict house sale price based on selected variables.\n"
    )

    # text based on README file - "Dataset Content & Business
    # Requirements" section
    st.info(
        f"**Project Terms & Jargon**\n"
        f"* An **Attributes** refers to the individual characteristics or"
        f" properties that describe the house.\n "
        f"* A **Correlation** measures the statistical relationship "
        f" between two variables. "
        f" It quantifies the degree to which changes in one variable "
        f" are associated with changes in another variable.\n "
        f"* A **Variable** represents a characteristic or property "
        f" that can take different values.\n "
        f"* An **ML** focuses on developing algorithms and models that "
        f" enable computer systems to learn from data and make predictions"
        f" or decisions without being explicitly programmed. \n "
        f"* **Hypothesis** is a testable statement about the "
        f" relationship between two or more variables or a proposed "
        f" explanation for some observed phenomenon.\n\n "
        f"**Project Dataset**\n"
        f"* The dataset represents a housing records from Ames, Iowa.\n "
        f"* Here is a link to the live dataset from "
        f" [Kaggle](https://www.kaggle.com/datasets/codeinstitute/housing-prices-data), "
        f" containing dataset of almost 1.5 thousand rows indicating house "
        f" profile of 23 variables such as Floor Area, Basement, Garage, "
        f" Kitchen, Lot, Porch, Wood Deck, Year Built and its "
        f" respective sale price for houses built between 1872 and 2010.\n "
        f"* Our **target** variable is the **'SalePrice'** which was "
        f" studied in correlation with the features to predict house price. "
    )

    # Link to README file, so the users can have access to full
    # project documentation
    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/Shida18719/heritage-housing-issues)."
    )

    # copied from README file - "Business Requirements" section
    st.success(
        f" A friend who lives in another country who has received inheritance "
        f" from deceased great grandfather in Ames, Iowa "
        f" is not familiar with the property prices in the USA "
        f" and fears that basing her estimates for property worth on her "
        f" current knowledge might lead to inaccurate appraisals.\n"
        f" We have been requested by our client to help in maximising the "
        f" sales price for the inherited properties. "
        f" We were provided with a public dataset house prices for Ames, Iowa,"
        f" and our client will like 2 business requirements be fulfilled.\n\n"
        f"** The 2 business requirements are:** \n\n"
        f"**1.** The client is interested in discovering how the house "
        f" attributes correlate with the sale price.\n "
        f" The client expects a dashboard for data visualisation of the "
        f" correlated variables against the sale price. \n\n"
        f"**2.** The client is interested in predicting the house sale price "
        f" from her four inherited houses and any other house in Ames, Iowa.\n"
        f" This will require the use of data analytics tools and machine "
        f" learning tasks that will allow the user to interact with variables "
        f" and enable user generate price predictions. "
    )

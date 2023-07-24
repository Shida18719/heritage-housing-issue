import streamlit as st


def page3_hypothesis_and_validation():

    st.write("### Hypothesis and Validation")

    st.info(
        f" We have used statistical analysis, eda evaluation,"
        f" correlation study and Predictive Power Score, "
        f" to help provide evidence and insights into the relationship,"
        f" allowing us draw a conclusions about the hypotheses"
        f" based on the results obtained.\n\n"
        f" Below is the **Process** and **Validation** "
        f" as to how these conclusions were achieved: \n\n"
    )

    # conclusions taken from "02 - Correlation Study" notebook
    st.success(
        "#### Hypothesis 1\n\n "
        f"The larger the size of a property "
        f" in square feet the higher the price\n\n"

        f"**Variable study**: Studying the correlations between the size "
        f" of the property (in square footage) and the sale value.\n "
        f" We observed if this feature has a significant positive"
        f" effect on house prices.\n\n"
        f"**Hypothesis validation**: Correct - The correlation study "
        f" at SalesPrice Correlation Notebook supports it. \n\n"
        f"**Validation process**: A custom functions where a combined "
        f" correlation (Pearson and Spearman) and Predictive Power Score (PPS)"
        f" analysis indicates that the size of the house "
        f" has a relatively high correlation to the house sale price.\n "
        f"* With a Pearson, Spearman and a PPS score indicating  "
        f" a strong to moderate predictive correlation with sales price.\n\n"
        f"* The scatter plots show where the 1stFlrSF', 'GarageArea', "
        f" 'GrLivArea and 'TotalBmntSF' have greater square footage, "
        f" the sale prices generally tend to be higher.  "
        f" This supports our first hypothesis. \n\n"

        "#### Hypothesis 2\n\n "
        f" The higher the overall quality rating of "
        f" the property, the higher the price \n\n"

        f"**Variable study**: Studying the correlations between the rating "
        f" of the property (in its quality) and the sale value, "
        f" we observed if this variable is a factor that influences price.\n\n"
        f"**Hypothesis validation**: Correct - The correlation study "
        f" at SalesPrice Correlation Notebook supports it.\n\n "
        f"**Validation process**: A custom functions, where a combined "
        f" correlation analysis (Pearson and Spearman)"
        f" and Predictive Power Score (PPS) analysis indicates that "
        f" high quality rating of the material and finish of the house "
        f" has a relatively high correlation to the house sale price.\n "
        f"* With a Pearson of 0.79 and Spearman score of 0.8, while a PPS "
        f" of 0.4 score indicate a strong correlation with sales price. \n"
        f"* The scatter plots shows the highest priced houses were ranked "
        f" of the highest Overall Quality. \n\n"

        "#### Hypothesis 3\n\n "
        f" We suspect the more recent the construction year "
        f" of the property, the higher the sale price.\n\n "

        f"**Variable study**: We studied data regarding the year houses "
        f"were constructed and observed "
        f" if it correlates with a higher price value.\n\n"
        f"**Hypothesis validation**: Correct - The correlation study "
        f" at SalesPrice Correlation Notebook supports it.\n\n "
        f"**Validation process**: A scatter plot observation shows, "
        f" a significant difference in the recently constructed property. "
        f" We assume there might be other factors that "
        f" could be attributed to price rise."
        f" This would make a good case for further investigations, as the "
        f" **GarageYrBult** has a strong correlation against the **YearBult**."
        f" This might have an indirect correlation with the sale price \n"
        f"* A spearman score of 0.65, while a PPS of 0.2 indicate "
        f" a moderate predictive correlation with sales price.\n"
        f"* The scatter plot, coupled with the Correlation and PPS analysis,"
        f" shows there is a price correlation in houses built after "
        f" 1980, and 2000, showing a positive correlation "
        f" with a higher selling price."
        f" These insights thus support our study. "
    )

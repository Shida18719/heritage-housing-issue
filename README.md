# Heritage Housing Issues

![]()

Heritage Housing Issues is a Data Analytics and Machine Learning Web App to predict the Sales of house price from a four inherited house and any other houses within the surrounding area of Ames.The project has been developed as part of a Portfolio Project of my Predictive Analytics studies, a Project-5 at Code Institute.

Link to deployed site:

---

## CONTENTS

- [Dataset](#dataset)
- [Business Requirements](#business-requirements)
- [Rationale to map the business requirements to the Data Visualizations and ML tasks](#rationale-to-map-the-business-requirements-to-the-data-visualizations-and-ml-tasks)
- [ML Business Case](#ml-business-case)
- [Hypothesis and Validation](#hypothesis-and-validation)
- [Dashboard Design](#dashboard-design)
- [Bugs](#bugs)
- [Deployment](#deployment)
- [Main Data Analysis and Machine Learning Libraries](#main-data-analysis-and-machine-learning-libraries)
- [Credits](#credits)
  - [Libraries and Packages](#libraries-and-packages)

<hr>

## How to use this repo

1. Use this template to create your GitHub project repo

1. Log into <a href="https://app.codeanywhere.com/" target="_blank" rel="noreferrer">CodeAnywhere</a> with your GitHub account.

1. On your Dashboard, click on the New Workspace button

1. Paste in the URL you copied from GitHub earlier

1. Click Create

1. Wait for the workspace to open. This can take a few minutes.

1. Open a new terminal and <code>pip3 install -r requirements.txt</code>

1. In the terminal type <code>pip3 install jupyter</code>

1. In the terminal type <code>jupyter notebook --NotebookApp.token='' --NotebookApp.password=''</code> to start the jupyter server.

1. Open port 8888 preview or browser

1. Open the jupyter_notebooks directory in the jupyter webpage that has opened and click on the notebook you want to open.

1. Click the button Not Trusted and choose Trust.

Note that the kernel says Python 3. It inherits from the workspace so it will be Python-3.8.12 as installed by our template. To confirm this you can use <code>! python --version</code> in a notebook code cell.

## Cloud IDE Reminders

To log into the Heroku toolbelt CLI:

1. Log in to your Heroku account and go to _Account Settings_ in the menu under your avatar.
2. Scroll down to the _API Key_ and click _Reveal_
3. Copy the key
4. In Gitpod, from the terminal, run `heroku_config`
5. Paste in your API key when asked

You can now use the `heroku` CLI program - try running `heroku apps` to confirm it works. This API key is unique and private to you so do not share it. If you accidentally make it public then you can create a new one with _Regenerate API Key_.

## Dataset Content

- The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
- The dataset has almost 1.5 thousand rows and represents housing records from Ames, Iowa, indicating house profile (Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and its respective sale price for houses built between 1872 and 2010.

| Variable      | Meaning                                                                 | Units                                                                                                                                                                   |
| :------------ | :---------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1stFlrSF      | First Floor square feet                                                 | 334 - 4692                                                                                                                                                              |
| 2ndFlrSF      | Second-floor square feet                                                | 0 - 2065                                                                                                                                                                |
| BedroomAbvGr  | Bedrooms above grade (does NOT include basement bedrooms)               | 0 - 8                                                                                                                                                                   |
| BsmtExposure  | Refers to walkout or garden level walls                                 | Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement                                                                       |
| BsmtFinType1  | Rating of basement finished area                                        | GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement |
| BsmtFinSF1    | Type 1 finished square feet                                             | 0 - 5644                                                                                                                                                                |
| BsmtUnfSF     | Unfinished square feet of basement area                                 | 0 - 2336                                                                                                                                                                |
| TotalBsmtSF   | Total square feet of basement area                                      | 0 - 6110                                                                                                                                                                |
| GarageArea    | Size of garage in square feet                                           | 0 - 1418                                                                                                                                                                |
| GarageFinish  | Interior finish of the garage                                           | Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage                                                                                                    |
| GarageYrBlt   | Year garage was built                                                   | 1900 - 2010                                                                                                                                                             |
| GrLivArea     | Above grade (ground) living area square feet                            | 334 - 5642                                                                                                                                                              |
| KitchenQual   | Kitchen quality                                                         | Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor                                                                                                        |
| LotArea       | Lot size in square feet                                                 | 1300 - 215245                                                                                                                                                           |
| LotFrontage   | Linear feet of street connected to property                             | 21 - 313                                                                                                                                                                |
| MasVnrArea    | Masonry veneer area in square feet                                      | 0 - 1600                                                                                                                                                                |
| EnclosedPorch | Enclosed porch area in square feet                                      | 0 - 286                                                                                                                                                                 |
| OpenPorchSF   | Open porch area in square feet                                          | 0 - 547                                                                                                                                                                 |
| OverallCond   | Rates the overall condition of the house                                | 10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor                                 |
| OverallQual   | Rates the overall material and finish of the house                      | 10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor                                 |
| WoodDeckSF    | Wood deck area in square feet                                           | 0 - 736                                                                                                                                                                 |
| YearBuilt     | Original construction date                                              | 1872 - 2010                                                                                                                                                             |
| YearRemodAdd  | Remodel date (same as construction date if no remodelling or additions) | 1950 - 2010                                                                                                                                                             |
| SalePrice     | Sale Price                                                              | 34900 - 755000                                                                                                                                                          |

---

## Business Requirements

As a good friend, you are requested by your friend, who has received an inheritance from a deceased great-grandfather located in Ames, Iowa, to help in maximising the sales price for the inherited properties.

Although your friend has an excellent understanding of property prices in her own state and residential area, she fears that basing her estimates for property worth on her current knowledge might lead to inaccurate appraisals. What makes a house desirable and valuable where she comes from might not be the same in Ames, Iowa. She found a public dataset with house prices for Ames, Iowa, and will provide you with that.

- 1 - The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.
- 2 - The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa.

---

## The rationale to map the business requirements to the Data Visualisations and ML tasks

- List your business requirements and a rationale to map them to the Data Visualisations and ML tasks.

---

## ML Business Case

- In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course.

---

## Hypothesis and Validation

**Hypothesis 1:**

- The size of a house, as measured by the square footage, is positively correlated with its sale price.
  - We were able to observe if this feature is a factor that influences price.

**Validation process:**

- This hypothesis suggests that larger houses tend to have higher sale prices.
- This hypothesis was tested and validated using statistical analysis and machine learning techniques to analysing the dataset of house feature and sale prices.
- We visualized the data using scatter plots, Correlations Study and Predictive Power Score (PPS) heatmaps to identify any initial trends or patterns.

**Interpretaion:**

- Analysis indicates that the size of the house has a relatively high correlation to the house sale price.

---

**Hypothesis 2:**

- "The rating of the house in quality has a significant positive effect on house prices."
  - We were able to observe if this feature is a factor that influences price.

**Validation process:**

- This hypothesis was tested and validated using statistical analysis and machine learning techniques to analysing the dataset of house feature and sale prices.
- We visualized the data using scatter plots, Correlations Study and Predictive Power Score (PPS) heatmaps to identify any initial trends or patterns.

**Interpretaion:**

- Analysis indicates that the quality of the house has a relatively high correlation to the house sale price.

---

**Hypothesis 3:**

- The most recently constructed house, has significant positive correlation with the sale price.
  - we were able to observe if this feature is a factor that influences price.

**Validation process:**

- This hypothesis was tested and validated using statistical analysis and machine learning techniques to analysing the dataset of house feature and sale prices.
- We visualized the data using scatter plots, Correlations Study and Predictive Power Score (PPS) heatmaps to identify any initial trends or patterns.

**Interpretaion:**

- Analysis indicates that the quality of the house has a relatively high correlation to the house sale price.

---

## Dashboard Design

- The dashboard is made up of 5 pages, and split into non-technical and technical user. The first three pages are non-technical, while the last two are technical.

### Page 1: Project Summary

- Summary
- Project Terms and Jargons
- Project Dataset
- Business Requirements

### Page 2: House Price Correlation Study

- Answers Business Requirements 1
  This page contains the following:
- Before the analysis, we knew we wanted this page to answer business requirement 1, but we couldn't know in advance which plots would need to be displayed.
- After data analysis, we agreed with stakeholders that the page will:
  - State business requirement 1
  - Checkbox: data inspection on customer base (display the number of rows and columns in the data, and display the first ten rows of the data)
  - Display the most correlated variables to sale price and the conclusions
  - Check boxes to display the heatmaps correlation analysis of individual variables against sell price.
  - Check boxes to display the scatterplots of correlated variables against sell price.

### Page 3: Project Hypothesis and Validation

- Before the analysis, we knew we wanted this page to describe each project hypothesis, the conclusions, and how we validated each. After the data analysis, we can report that:

1 - The larger the size of a property in square feet the higher the price

2 - The higher the overall quality rating of the property, the higher the price

3 - We suspect the more recent the construction year of the property, the higher the sale price.

- Hypothesis validation: Correct. The correlation study at SalesPrice Correlation Notebook supports them.

### Page 4: Predict House Price

- Answers Business requirement 2 - "The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa."
- Table displaying the profile of the four inherited houses and the house attributes.
- Table displaying the 5 most relevant features used by the ML model for predicting house sale price.
- The estimated individual value predicted for the 4 inherited houses
- Predict House Prices in Ames, Iowa - 2 blocks explaining the selected features used by the ML for predicting house price.
- Set of widgets inputs, which relates to the house profile. Each set of inputs is related to a given ML task to predict house prices.
- "Run predictive analysis" button that serves the house price data to our ML pipelines, and predicts sale price.

### Page 5: ML Prediction Metrics

- ML Pipeline Performance - block explaining the steps taken to complete the ml pipeline
- Considerations and conclusions after the pipeline is trained
- Present ML pipeline steps
- Feature importance plot
- Pipeline performance
- Predicted Price versus actual Sale Price Scatterplot

---

## Unfixed Bugs

- You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a big variable to consider, paucity of time and difficulty understanding implementation is not valid reason to leave bugs unfixed.

## Deployment

### Heroku

- The App live link is: https://YOUR_APP_NAME.herokuapp.com/
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

- Here you should list the libraries you used in the project and provide example(s) of how you used these libraries.

## Credits

- In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism.
- You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)
- SkLearn(https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
  -PPS interpretation (https://github.com/8080labs/ppscore/issues/39)

### Media

- The Favicon (https://twemoji-cheatsheet.vercel.app/)

## Acknowledgements

I would like to show my sincere appreciation to the following people who have helped me along the way in completing this project:

- My family, for their understanding, for being such an important part of my life, and for making every day a little bit brighter.
- Course provider - Code Institute.
- The slack community, for always being there.

# HW1_Box_Office_Revenue_Prediction

This code is for an assignment for the course: Laboratory Analysis and Presentation of Data. 

In this project we used data about movies in order to predict the global revenue of the movies in the theaters. 
The data includes 5215 train samples, and 1738 test samples. 

This repository includes:
1. environment.yml - used for creating a suitable environment. 
2. predict.py - this code loads the trained models and the given test data and calculates the predictions for  the target: "revenue". The resulted predictions are exported to the same directory as a CSV file with 2 columns: the predictions and the id of the sample. 
3. f_selected_features.sav - pickle file that contains a list of selected explanatory features for the model (used in predict.py). 
4. f_tuned_et_best.sav - pickle file with ExtraTreesRegressor trained model. 
5. tuned_f.zip - zip file that includes a pickle file with RandomForestRegressor traind model. This was uploaded as a zip file since the pickle file was too large for uploading it to the repository. predict.py unzip this file and extract the model from it. 
6. HW1_report.ipynb - A code and report for this project. The report includes: EDA, feature engineering, training and evaluating models, choosing and ensembling models, and saving the final models. 
7. instructions.pdf - instructions for the assignment. 


In order to run the code, please run predict.py 

Enjoy, 
Almog and Inbal 

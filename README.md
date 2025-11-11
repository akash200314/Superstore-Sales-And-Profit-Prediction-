🧠 Superstore Sales and Profit Prediction using Machine Learning

📋 Overview
This project builds a machine learning regression model to predict the Profit of a retail store based on Sales and Discount data from the Superstore dataset.  
It helps identify how pricing and discounts affect profitability and assists business teams in making data-driven decisions.

🎯 Objectives
- Predict Profit using key sales features like Sales and Discount.
- Automate data preprocessing, feature scaling, model training, and evaluation.
- Generate visual insights (Feature Importance and Actual vs Predicted plots).
- Save the trained model and predicted outputs for future use.


🧰 Tech Stack

 Language - Python 
 Data Processing - pandas, numpy 
 Machine Learning - scikit-learn 
 Model - Random Forest Regressor 
 Visualization - matplotlib 
 File Handling - openpyxl, joblib 


📂 Project Structure

📦 superstore-ml-regression
📜 superstore.xlsx # Dataset (Superstore data)
📜 superstore_regression_with_plots.py # Main ML script
📜 predictions.xlsx # Output predictions (Actual vs Predicted)
📜 trained_pipeline.joblib # Saved ML model
📊 feature_importance.png # Feature importance bar chart
📊 actual_vs_predicted.png # Actual vs Predicted scatter plot
📄 README.md # Project documentation

Outputs

📈 Model Metrics: MAE, RMSE, and R² printed on console.
📊 Visualizations:
feature_importance.png — shows how much each feature contributes.
actual_vs_predicted.png — visual comparison of actual vs predicted profit.
📁 Files Generated:
predictions.xlsx — Excel with actual and predicted profit values.
trained_pipeline.joblib — saved ML model for reuse.
📊 Model Performance
Metric	Description
MAE	     -  Measures average prediction error in monetary terms
RMSE	    - Penalizes larger errors more heavily
R² Score   - Indicates how well the model explains the variance in Profit

🔍 Insights
Discount and Sales strongly influence Profit margins.
The Random Forest Regressor provided robust performance with low error rates.
Visualization shows close alignment between actual and predicted values, indicating good generalization.
<img width="590" height="390" alt="download" src="https://github.com/user-attachments/assets/8b458eea-9b42-4e5d-a231-2c7e404fd346" />

<img width="590" height="590" alt="download" src="https://github.com/user-attachments/assets/8ee1b15e-33de-4dde-8c46-2d2f9cb066b2" />

🚀 Future Enhancements

Include more features (e.g., Region, Category, Quantity) for better accuracy.
Experiment with other ML models (XGBoost, Linear Regression, etc.).
Deploy the model using Flask or Streamlit for real-time predictions.
Add interactive dashboards using Power BI or Plotly.

👨‍💻 Author

Akash Nalawade
Data Analyst | Machine Learning Enthusiast
📧 [akashnalawade2003@gmail.com]
🔗 [https://www.linkedin.com/in/akash-nalawade-20aan]

🏁 Conclusion
This project demonstrates the complete pipeline of a machine learning regression task — from data preprocessing to model evaluation and visualization — applied to real-world sales data for actionable business insights.

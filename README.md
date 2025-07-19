#Framingham Heart Disease Risk Prediction:
This repository contains a machine learning project that uses the **Framingham Heart Study dataset** to predict whether an individual will develop **coronary heart disease (CHD)** within the next 10 years. The prediction is made using a **logistic regression** model based on key health metrics.

File Structure:
├── framingham.csv    # Dataset file
├── main.py           # Main script for model and visualization
├── README.md         # Project documentation

#Project Objectives:
- Clean and preprocess the dataset (handle missing values, normalize features).
- Explore and visualize the distribution of key variables.
- Train a logistic regression model for binary classification.
- Evaluate the model's performance using accuracy, confusion matrix, and classification report.

#Dataset Overview:
The dataset comes from the [Framingham Heart Study](https://www.framinghamheartstudy.org/) and contains anonymized health data.

**Target variable:**
- `TenYearCHD`: 1 if the patient is likely to develop CHD in 10 years, otherwise 0.

**Selected Features used for training:**
- `age`: Age of the patient
- `Sex_male`: Gender (1 for male, 0 for female)
- `cigsPerDay`: Number of cigarettes smoked per day
- `totChol`: Total cholesterol level
- `sysBP`: Systolic blood pressure
- `glucose`: Glucose level

#Technologies Used:
- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (for preprocessing, modeling, and evaluation)

#Exploratory Data Analysis:
The project includes visualizations such as:
- Countplot of CHD occurrences
- Histogram showing CHD frequency distribution
- Heatmap of feature correlations
These plots help to understand the data distribution and detect potential patterns.

#Model Training:
- The logistic regression model was trained on 70% of the data using `train_test_split`.
- Features were scaled using `StandardScaler` to improve model convergence.
- Accuracy and confusion matrix were used for performance evaluation.

#Results:
- Model Accuracy: Achieved on the test set.
- Confusion Matrix: Visualized using a heatmap for better clarity.
- Classification Report: Includes precision, recall, F1-score, and support for both classes.

#How to Run the Code:
1. Clone this repository:
    git clone https://github.com/LakshyaPRJ/framingham-heart-disease-prediction.git
    cd framingham-heart-disease-prediction
2. Make sure you have all dependencies installed:
    pip install pandas numpy matplotlib seaborn scikit-learn
3. Place the framingham.csv dataset in the project directory.
4. Run the script:
    python heart_disease_predict.py
   

# Medical-Insurance-Project1
🧠 Predicting Income Brackets Using Random Forest Author: Yasmine Naser Target Roles: Machine Learning Engineer | AI Engineer

📌 Project Overview This project applies supervised learning techniques to classify whether an individual's income exceeds $50K based on census data. It demonstrates the ability to build and tune machine learning models, perform feature engineering, and evaluate model performance efficiently.

The primary goal is to design a robust pipeline using a Random Forest Classifier, optimize its parameters to prevent overfitting, and extract interpretable insights through feature importance — all aligned with core responsibilities of ML and AI engineering roles.

🎯 Objectives Develop a predictive model using Random Forest

Implement scalable feature encoding and engineering

Tune model hyperparameters for generalization

Visualize performance trends over depth tuning

Quantify the impact of features on model output

📂 Dataset Source: UCI Machine Learning Repository – Adult Income Dataset

Rows: ~32,000

Columns: 14 features + 1 target

Target: income (binary: >50K or <=50K)

🛠️ Tools & Stack Programming: Python

Libraries: scikit-learn, pandas, numpy, matplotlib, seaborn

Model: RandomForestClassifier (ensemble method)

Evaluation: Accuracy, Feature Importance

⚙️ Workflow

Data Preprocessing Whitespace stripping from categorical fields
One-hot encoding for categorical variables

Binary transformation of target (income)

Model Training Split data using train_test_split
Trained baseline Random Forest with default parameters

Evaluated accuracy on both train and test data

Hyperparameter Tuning Tuned max_depth from 1 to 25
Tracked accuracy trends to identify optimal depth

Plotted results for visual comparison

Feature Engineering Created education_bin to group higher education levels
Created native_country_bin to simplify country feature

Retrained model with new engineered features

Feature Importance Extracted and ranked top features from best-fit model
Identified age, capital-gain, education_bin, and hours-per-week as top predictors

📊 Results Best Accuracy (Test Set): ~85%

Most Influential Features:

age

capital-gain

education_bin

hours-per-week

native_country_bin

✅ Skills Demonstrated Machine learning model selection & tuning

Supervised classification using ensemble methods

Feature transformation & binary encoding

Model evaluation and generalization control

Feature impact interpretation and analysis

Applied software engineering workflows in ML

🚀 Future Improvements Add cross-validation for better generalization metrics

Integrate pipeline into deployment-ready module (Flask/FastAPI)

Explore model compression for production efficiency

Compare with other ensemble methods like Gradient Boosting

👩‍💻 About Me I'm Yasmine Naser, an aspiring Machine Learning and AI Engineer with a strong interest in scalable model design, structured experimentation, and solving complex prediction problems with optimized, interpretable solutions.

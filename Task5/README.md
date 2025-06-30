# Task 5 - Decision Trees and Random Forests 🧠🌳

## ⭐ Objective
The goal of this task is to build machine learning models using **Decision Trees** and **Random Forests** to predict whether a person has heart disease or not based on their health attributes.

---

## 📚 Dataset
- **Dataset Used:** Heart Disease Dataset from Kaggle
- **Features:** 
  - age
  - sex
  - chest pain type (cp)
  - resting blood pressure (trestbps)
  - serum cholestoral (chol)
  - fasting blood sugar (fbs)
  - resting ECG (restecg)
  - maximum heart rate achieved (thalach)
  - exercise induced angina (exang)
  - oldpeak (ST depression)
  - slope of peak exercise ST segment (slope)
  - number of major vessels (ca)
  - thal (thalassemia)

- **Target:** 
  - 0 = No Heart Disease
  - 1 = Heart Disease

---

## 🛠 Tools & Libraries Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

---

## 🚀 What I Did
1. **Data Preprocessing:** 
   - Loaded dataset, handled missing values.

2. **Decision Tree Classifier:**
   - Trained a decision tree classifier.
   - Visualized the tree.
   - Accuracy: **98.5%**
   - Also trained a **limited-depth tree** to observe **overfitting**, accuracy dropped to **80%**.

3. **Random Forest Classifier:**
   - Trained a random forest (an ensemble of multiple trees).
   - Accuracy: **98.5%**, with better generalization.
   - Observed feature importance.

4. **Cross-Validation:**
   - Checked model performance with cross-validation:
     - Decision Tree: **100%**
     - Random Forest: **99.7%**

5. **Evaluation:**
   - Printed classification reports.
   - Visualized decision boundaries and tree structure.

---

## 🖼 Screenshots
Screenshots of outputs are available in the `screenshots/` folder.

---

## 📊 Results Summary

| Model                        | Accuracy  | Cross-Validation |
|------------------------------|-----------|------------------|
| Decision Tree (Full)         | 98.5%     | 100%             |
| Decision Tree (Limited Depth)| 80%       | —                |
| Random Forest                | 98.5%     | 99.7%            |

---

## 📌 Learnings
- Decision Trees are simple but can overfit easily.
- Limiting tree depth reduces overfitting but may lower accuracy.
- Random Forest solves overfitting by combining multiple trees (Bagging).
- Learned about feature importance and how to interpret it.

---

## 👨‍💻 How to Run
```bash
python3 DecisionTree_RandomForest.py


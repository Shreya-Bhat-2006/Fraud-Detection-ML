# Fraud Detection System

##  Project Aim

To build a machine learning-based fraud detection system that helps banks reduce financial loss by identifying fraudulent transactions before they are processed.

---

##  Dataset Overview


Dataset used is from:   https://www.kaggle.com/datasets/ealaxi/paysim1

The dataset contains records of financial transactions. Each transaction has:

* **Amount transferred**
* **Sender account balances**
* **Receiver account balances**
* **Transaction type** (e.g., `TRANSFER`, `CASH_OUT`)
* **Label indicating fraud** (`isFraud` = 1 means fraud)

However, only **a very small number of transactions are fraud**, which creates **class imbalance**.

---

##  Handling Data Imbalance

Fraud cases are rare, so the dataset is **imbalanced**.
To solve this, I used **SMOTE (Synthetic Minority Oversampling Technique)**:

* SMOTE generates *synthetic fraudulent samples*
* Balances the number of fraud and non-fraud transactions

This helps the model **not ignore the fraud class**.

---

##  Feature Engineering

To improve detection accuracy, I created **new features**:

| Feature Name              | Meaning                                                          |
| ------------------------- | ---------------------------------------------------------------- |
| `amount_to_balance_ratio` | Checks if transaction amount is large compared to sender balance |
| `receiver_is_new`         | Detects if receiver account has zero balance before transaction  |
| `large_transaction`       | Flags transactions above a high amount threshold                 |

I also converted **transaction type** into **one-hot encoded features** using `pd.get_dummies()`.

---

##  Model Used

I trained a **Random Forest Classifier** because:

* It handles large datasets well
* Works effectively on non-linear relationships
* Reduces risk of overfitting

### Model Performance Evaluation

I used:

* **Confusion Matrix**
* **Classification Report (Precision, Recall, F1-score)**

The final model was saved using:

```python
joblib.dump(model, "fraud_model.sav")
joblib.dump(feature_list, "model_features.sav")
```

---

##  Rule-Based Risk System (Additional Safety Layer)

Along with machine learning prediction, I added **manual logic rules** to evaluate risk:

* Very high transaction amount
* Sender has very low or zero balance
* Receiver account is new
* High-risk transaction types (`TRANSFER`, `CASH_OUT`)

The model risk score and rule-based score are **combined** to produce the final fraud probability.

This makes the system **much more reliable**.

---

##  Streamlit UI

A **Web UI** was built using Streamlit to make fraud detection easy for users.
The user inputs transaction details → system outputs:

* Model Prediction Score
* Rule-Based Score
* Final Risk Score
* Recommendation (Block / Hold / Safe)

---

##  Running the Project

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Run Streamlit App

```
streamlit run App/app.py
```

---

##  Final Output

The system provides:

* Fraud probability score
* Reasons for risk
* Transaction recommendation

This helps banks **prevent fraudulent transactions in real-time**.

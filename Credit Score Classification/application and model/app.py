import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


# Load the trained model and scaler
with open('fdm_model.pickle', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
    
st.image("https://img.freepik.com/free-vector/people-near-bank-building_74855-4455.jpg", use_column_width=True)

# Define encoders
ordinal_encoder_credit_mix = OrdinalEncoder(categories=[['Bad', 'Standard', 'Good']])
le_score = LabelEncoder()

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Predict Manually", "Batch Prediction"])

if page == "Predict Manually":
    # Original code for manual input-based prediction
    st.title("Customer Credit Score Predictor")
    st.write("Enter the following details to predict your credit score:")

    # Input fields for necessary features
    occupation = st.selectbox('Occupation', ['Accountant', 'Architect', 'Developer', 'Doctor', 'Engineer', 'Entrepreneur',
                                             'Journalist', 'Lawyer', 'Manager', 'Mechanic', 'Media_Manager', 'Musician',
                                             'Scientist', 'Teacher', 'Writer'])
    Annual_Income = st.number_input('Annual Income (in USD)', min_value=0.0, format="%.2f")
    num_bank_accounts = st.number_input('Number of Bank Accounts', min_value=0, step=1)
    num_credit_card = st.number_input('Number of Credit Cards', min_value=0, step=1)
    Interest_Rate = st.number_input('Interest Rate (%)', min_value=0.0, format="%.2f")
    Num_of_Loan = st.number_input('Number of Loans', min_value=0, step=1)
    delay_from_due_date = st.slider('Delay from Due Date (days)', 0, 60, 0)
    num_of_delayed_payment = st.number_input('Number of Delayed Payments', min_value=0, step=1)
    num_credit_inquiries = st.number_input('Number of Credit Inquiries', min_value=0, step=1)

    credit_mix = st.selectbox('Credit Mix', ['Bad', 'Standard', 'Good'])


    payment_of_min_amount = st.selectbox('Payment of Minimum Amount', ['yes', 'no', 'NM'])
    payment_of_min_amount = {'yes': 2, 'no': 1, 'NM': 0}[payment_of_min_amount]

    outstanding_debt = st.number_input('Outstanding Debt (in USD)', min_value=0.0, format="%.2f")
    credit_history_age = st.number_input('Credit History Age (Months)', min_value=0, step=1)

    total_num_accounts = num_bank_accounts + num_credit_card

    credit_mix_encoded = ordinal_encoder_credit_mix.fit_transform(np.array([[credit_mix]]))[0][0]

    occupation_list = ['Occupation_Accountant', 'Occupation_Architect', 'Occupation_Developer', 'Occupation_Doctor',
                       'Occupation_Engineer', 'Occupation_Entrepreneur', 'Occupation_Journalist', 'Occupation_Lawyer',
                       'Occupation_Manager', 'Occupation_Mechanic', 'Occupation_Media_Manager', 'Occupation_Musician',
                       'Occupation_Scientist', 'Occupation_Teacher', 'Occupation_Writer']

    occupation_encoded = [1 if f'Occupation_{occupation}' == occ else 0 for occ in occupation_list]

    numerical_features = np.array([[Annual_Income, Interest_Rate, Num_of_Loan, delay_from_due_date, num_of_delayed_payment, 
                                    num_credit_inquiries, outstanding_debt, credit_history_age, total_num_accounts]])

    numerical_features_scaled = scaler.transform(pd.DataFrame(numerical_features, columns=[
        'Annual_Income', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 
        'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_History_Age', 'Total_Num_Accounts']))

    final_input = np.concatenate([numerical_features_scaled[:, :6], np.array([[credit_mix_encoded]]), 
                                  numerical_features_scaled[:, 6:8], np.array([[payment_of_min_amount]]), 
                                  np.array(occupation_encoded).reshape(1, -1), 
                                  numerical_features_scaled[:, -1:]], axis=1)

    # st.write(final_input.shape)

    if st.button('Predict Credit Score'):
        prediction = model.predict(final_input)
        credit_score_mapping = {0: 'Poor', 1: 'Standard', 2: 'Good'}
        st.success(f'Predicted Credit Score: {credit_score_mapping.get(prediction[0], "Unknown")}')

    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)  # Use TreeExplainer for Random Forest
    shap_values = explainer(final_input)

 
    feature_names = [
        'Annual_Income', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 
        'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt', 'Credit_History_Age', 
        'Payment_of_Min_Amount'] + occupation_list + ['Total_Num_Accounts']

    st.write(f"Number of features: {len(feature_names)}")
    class_names = {0: 'Poor', 1: 'Standard', 2: 'Good'}

    class_names = {0: 'Poor', 1: 'Standard', 2: 'Good'}

    # Plot SHAP values
    for i in range(shap_values.shape[2]):
        class_name = class_names.get(i, f"Class {i}")
        st.subheader(f"SHAP values for {class_name}")
        
        # Create a new figure for each SHAP plot
        fig, ax = plt.subplots()  
        shap.summary_plot(shap_values[..., i], final_input, feature_names=feature_names, show=False)
        st.pyplot(fig)  # Pass the figure to Streamlit's pyplot


elif page == "Batch Prediction":
    
    st.title("Customer Credit Score Predictor")
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:", data.head())

        # Filter necessary columns
        required_columns = ['Customer_ID', 'Name', 'Occupation', 'Annual_Income', 'Num_Bank_Accounts', 
                            'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 
                            'Num_of_Delayed_Payment', 'Num_Credit_Inquiries', 'Credit_Mix', 
                            'Outstanding_Debt', 'Credit_History_Age', 'Payment_of_Min_Amount']
        
        occupation_list = ['Occupation_Accountant', 'Occupation_Architect', 'Occupation_Developer', 'Occupation_Doctor',
                       'Occupation_Engineer', 'Occupation_Entrepreneur', 'Occupation_Journalist', 'Occupation_Lawyer',
                       'Occupation_Manager', 'Occupation_Mechanic', 'Occupation_Media_Manager', 'Occupation_Musician',
                       'Occupation_Scientist', 'Occupation_Teacher', 'Occupation_Writer']

        if all(col in data.columns for col in required_columns):
            # Store prediction results
            results = []

            # Preprocessing for each row in the CSV
            for index, row in data.iterrows():
                Annual_Income = row['Annual_Income']
                num_bank_accounts = row['Num_Bank_Accounts']
                num_credit_card = row['Num_Credit_Card']
                Interest_Rate = row['Interest_Rate']
                Num_of_Loan = row['Num_of_Loan']
                delay_from_due_date = row['Delay_from_due_date']
                num_of_delayed_payment = row['Num_of_Delayed_Payment']
                num_credit_inquiries = row['Num_Credit_Inquiries']
                credit_mix = row['Credit_Mix']
                occupation = row['Occupation']
                outstanding_debt = row['Outstanding_Debt']
                credit_history_age = row['Credit_History_Age']
                payment_of_min_amount = row['Payment_of_Min_Amount']

                total_num_accounts = num_bank_accounts + num_credit_card
                
                payment_of_min_amount = {'Yes': 2, 'No': 1, 'NM': 0}.get(payment_of_min_amount, 0)

                credit_mix_encoded = ordinal_encoder_credit_mix.fit_transform(np.array([[credit_mix]]))[0][0]

                occupation_encoded = [1 if f'Occupation_{occupation}' == occ else 0 for occ in occupation_list]

                numerical_features = np.array([[Annual_Income, Interest_Rate, Num_of_Loan, delay_from_due_date, 
                                                num_of_delayed_payment, num_credit_inquiries, outstanding_debt, 
                                                credit_history_age, total_num_accounts]])

                numerical_features_scaled = scaler.transform(pd.DataFrame(numerical_features, columns=[
                    'Annual_Income', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 
                    'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_History_Age', 'Total_Num_Accounts']))

                final_input = np.concatenate([numerical_features_scaled[:, :6], np.array([[credit_mix_encoded]]), 
                                              numerical_features_scaled[:, 6:8], np.array([[payment_of_min_amount]]), 
                                              np.array(occupation_encoded).reshape(1, -1), 
                                              numerical_features_scaled[:, -1:]], axis=1)

                prediction = model.predict(final_input)
                credit_score_mapping = {0: 'Poor', 1: 'Standard', 2: 'Good'}
                predicted_score = credit_score_mapping.get(prediction[0], "Unknown")

              
                results.append({
                    'Customer ID': row['Customer_ID'],
                    'Name': row['Name'],
                    'Predicted Credit Score': predicted_score
                })

            results_df = pd.DataFrame(results)
            st.write("Prediction Results:")
            st.dataframe(results_df)
        else:
            st.error("CSV file does not contain the required columns.")

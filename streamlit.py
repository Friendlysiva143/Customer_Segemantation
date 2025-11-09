import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from scipy import stats
import os
# ==========================
# Load the saved components
# ==========================
if not os.path.exists("scaler.pkl"):
    st.error("❌'scaler.pkl' file not found in the current directory.")
    st.stop()
else:
    scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")             # PCA
model = joblib.load("final_model.pkl") # Your final clustering model (KMeans or GMM)

# ==========================
# Streamlit App UI
# ==========================
st.set_page_config(page_title="Customer Segmentation", layout="centered")

st.title("🎯 Market Customer Segmentation")
st.write("Upload a dataset or enter customer details to find their cluster group.")

# Sidebar menu
option = st.sidebar.radio("Choose Input Method", ["📁 Upload CSV File", "🧍 Enter Single Customer"])

# ==========================
#  Option A: Upload CSV
# ==========================
if option == "📁 Upload CSV File":
    # ------------------------------------------
    # File Upload
    # ------------------------------------------
    uploaded_file = st.file_uploader("Upload customer CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read uploaded CSV
        df = pd.read_csv(uploaded_file)

        st.write("### Preview of Uploaded Data")
        st.dataframe(df.head())

        # ------------------------------------------
        #  Handle Missing Values
        # ------------------------------------------
        st.write("### Handling Missing Values...")

        # Separate numeric and categorical columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        st.write(num_cols)

        # Fill missing numeric values with mean
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

        # Fill missing categorical values with mode
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

        st.success("✅ Missing values handled successfully!")

        # ------------------------------------------
        #  Apply Transformations (as used in training)
        # ------------------------------------------
        st.write("### Applying Transformations...")

        # Purchase amount transformations
        df['MntWines_transformed'] = np.sqrt(df['MntWines'])
        df['MntFruits_transformed'] = np.log1p(np.where(df['MntFruits'] < 0, 0, df['MntFruits']))
        df['MntMeatProducts_transformed'] = np.log1p(np.where(df['MntMeatProducts'] < 0, 0, df['MntMeatProducts']))
        df['MntFishProducts_transformed'] = np.log1p(np.where(df['MntFishProducts'] < 0, 0, df['MntFishProducts']))
        df['MntSweetProducts_transformed'] = np.log10(np.where(df['MntSweetProducts'] <= 0, 1, df['MntSweetProducts']))

        fitted_lambda_gold = 0.11350711836273918
        df['MntGoldProds_transformed'] = np.where(
            fitted_lambda_gold != 0,
            (np.power(df['MntGoldProds'] + 1, fitted_lambda_gold) - 1) / fitted_lambda_gold,
            np.log(df['MntGoldProds'] + 1)
        )

        # Purchase behavior transformations
        fitted_lambda_deals = -0.4871850676344033
        df['NumDealsPurchases_transformed'] = np.where(
            fitted_lambda_deals != 0,
            (np.power(df['NumDealsPurchases'] + 1, fitted_lambda_deals) - 1) / fitted_lambda_deals,
            np.log(df['NumDealsPurchases'] + 1)
        )

        df['NumWebPurchases_transformed'] = np.sqrt(df['NumWebPurchases'])
        df['NumCatalogPurchases_transformed'] = np.sqrt(df['NumCatalogPurchases'])
        df['NumStorePurchases_transformed'] = np.log1p(np.where(df['NumStorePurchases'] < 0, 0, df['NumStorePurchases']))
        df['NumWebVisitsMonth_transformed'] = df['NumWebVisitsMonth']

        #  Define the mapping dictionary
        marital_status_map = {
            "Together": 2,
            "Married": 2,
            "Single": 1,
            "Divorced": 1,
            "Widow": 1,
            "Alone": 1,
            "Absurd": 2,
            "YOLO": 2
        }

        #  Apply mapping to your uploaded CSV
        if 'Marital_Status' in df.columns:
            df['Marital_Status'] = df['Marital_Status'].map(marital_status_map)

            # If any unmapped values exist (NaN after mapping), handle them
            df['Marital_Status'] = df['Marital_Status'].fillna(1)  # or another default value
        else:
            st.warning("⚠️ 'Marital_Status' column not found in the uploaded CSV.")


        # ------------------------------------------
        #  Feature Engineering
        # ------------------------------------------
        # FamilySize recomputation (if not present)
        if 'FamilySize' not in df.columns and {'Kidhome', 'Teenhome', 'Marital_Status'}.issubset(df.columns):
            df['FamilySize'] = df['Kidhome'] + df['Teenhome'] + df['Marital_Status']

        # Education mapping
        edu_map = {"Basic": 1, "2n Cycle": 2, "Graduation": 3, "Master": 4, "PhD": 5}
        if 'Education' in df.columns:
            df['Education_Ordinal'] = df['Education'].map(edu_map)

        # Campaign total if not present
        if 'campaign' not in df.columns and {'AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5'}.issubset(df.columns):
            df['campaign'] = df[['AcceptedCmp1','AcceptedCmp2','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5']].sum(axis=1)
        
        # List of categories you want to one-hot encode
        marital_categories = ["Married", "Single", "Together", "Widow"]

        # Initialize columns to 0
        for category in marital_categories:
            df[f"Marital_Status_{category}"] = 0

        # Set 1 for the matching category
        for category in marital_categories:
            df.loc[df['Marital_Status'] == category, f"Marital_Status_{category}"] = 1


        # ------------------------------------------
        #  Define Final Feature Columns
        # ------------------------------------------
        feature_cols = [
            'Income', 'Response', 'Age', 'FamilySize',
            'MntWines_transformed', 'MntFruits_transformed', 'MntMeatProducts_transformed',
            'MntFishProducts_transformed', 'MntSweetProducts_transformed', 'MntGoldProds_transformed',
            'NumDealsPurchases_transformed', 'NumWebPurchases_transformed',
            'NumCatalogPurchases_transformed', 'NumStorePurchases_transformed',
            'NumWebVisitsMonth_transformed', 'Customer_Since_Days',
            'Education_Ordinal', 'Marital_Status_Married', 'Marital_Status_Single',
            'Marital_Status_Together', 'Marital_Status_Widow', 'campaign'
        ]

        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            st.warning(f"⚠️ Missing columns in uploaded file: {missing_cols}")
        else:
        
            # ------------------------------------------
            # Transform & Predict
            # ------------------------------------------
            scaled = scaler.transform(df[feature_cols])
            pca_data = pca.transform(scaled)
            df['Cluster'] = model.predict(pca_data)

            st.success("✅ Clusters predicted successfully!")
            st.dataframe(df[['Cluster'] + feature_cols].head())

            # ------------------------------------------
            #  Optional: Cluster Interpretation
            # ------------------------------------------
            st.info("""
            🧭 **Cluster Interpretation:**
            - Cluster 0 → High-Value Loyal Customers  
            - Cluster 1 → Budget-Conscious Shoppers  
            - Cluster 2 → Family-Oriented Moderate Spenders  
            - Cluster 3 → Inactive or Lost Customers  
            - Cluster 4 → New or Potential High-Spenders  
            """)

            # ------------------------------------------
            #  Download Results
            # ------------------------------------------
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results CSV", csv, "clustered_customers.csv", "text/csv")
        
# ==========================
#  Option B: Single Customer Input
# ==========================
elif option == "🧍 Enter Single Customer":
    st.write("### Enter Customer Details")

    #  Replace these input fields with your actual feature names
    Income = st.number_input("Income", min_value=0, max_value=200000, step=1)

    Response_value=st.selectbox("Response",['Positive','Negative'])
    Response=0 if Response_value=='Negative' else 1
    st.write("Response value:",Response)
    year_birth=st.number_input("year of birth",min_value=1925,max_value=datetime.now().year)
    age = datetime.now().year-year_birth
    Age=18 if age < 18 else 100 if age>100 else age
    Kidhome = st.number_input("Number of Kids at Home", min_value=0, step=1)
    Teenhome = st.number_input("Number of Teens at Home", min_value=0, step=1)
    Marital_Status_value = st.selectbox(
        "Marital Status",
        ["Single", "Together", "Married", "Divorced", "Widow", "Alone", "Absurd", "YOLO"]
    )
    # Map display values to numeric codes
    marital_status_map = {
        "Together": 2,
        "Married": 2,
        "Single": 1,
        "Divorced": 1,
        "Widow": 1,
        "Alone": 1,
        "Absurd": 2,
        "YOLO": 2
    }

    # Get numeric value
    Marital_Status = marital_status_map[Marital_Status_value]

    st.write("Selected Marital Status:", Marital_Status_value)
    st.write("Mapped Value for Model:", Marital_Status)

    FamilySize = Kidhome+Teenhome+Marital_Status
    # Spending columns
    MntWines = st.number_input("Spent on Wines", min_value=0, step=1)
    MntFruits = st.number_input("Spent on Fruits", min_value=0, step=1)
    MntMeatProducts = st.number_input("Spent on Meat Products", min_value=0, step=1)
    MntFishProducts = st.number_input("Spent on Fish Products", min_value=0, step=1)
    MntSweetProducts = st.number_input("Spent on Sweet Products", min_value=0, step=1)
    MntGoldProds = st.number_input("Spent on Gold Products", min_value=0, step=1)
    # transformed spending columns 
    MntWines_transformed=np.sqrt(MntWines)
    # Safely handle negatives and NaNs before transformations
    MntFruits = np.where(np.isnan(MntFruits) | (MntFruits < 0), 0, MntFruits)
    MntMeatProducts = np.where(np.isnan(MntMeatProducts) | (MntMeatProducts < 0), 0, MntMeatProducts)
    MntFishProducts = np.where(np.isnan(MntFishProducts) | (MntFishProducts < 0), 0, MntFishProducts)
    MntSweetProducts = np.where(np.isnan(MntSweetProducts) | (MntSweetProducts <= 0), 1, MntSweetProducts)
    # Note: log10(0) or log10(negative) = -inf, so we replace 0 or negative values with 1 before transforming.

    # Apply transformations
    MntFruits_transformed = np.log1p(MntFruits)
    MntMeatProducts_transformed = np.log1p(MntMeatProducts)
    MntFishProducts_transformed = np.log1p(MntFishProducts)
    MntSweetProducts_transformed = np.log10(MntSweetProducts)

    fitted_lambda = 0.11350711836273918  # used the same lambda that was fit on training data
    # Box-Cox transformation manually
    if fitted_lambda != 0:
        MntGoldProds_transformed = (np.power(MntGoldProds + 1, fitted_lambda) - 1) / fitted_lambda
    else:
        MntGoldProds_transformed = np.log(MntGoldProds + 1)
    
    # Purchase behavior
    NumDealsPurchases = st.number_input("Number of Deals Purchases", min_value=0, step=1)
    NumWebPurchases = st.number_input("Number of Web Purchases", min_value=0, step=1)
    NumCatalogPurchases = st.number_input("Number of Catalog Purchases", min_value=0, step=1)
    NumStorePurchases = st.number_input("Number of Store Purchases", min_value=0, step=1)
    NumWebVisitsMonth = st.number_input("Number of Web Visits per Month", min_value=0, step=1)
    
    # transformed purchase behaviour
    fitted_lambda=-0.4871850676344033
    # Box-Cox transformation manually
    if fitted_lambda != 0:
        NumDealsPurchases_transformed = (np.power(NumDealsPurchases + 1, fitted_lambda) - 1) / fitted_lambda
    else:
        NumDealsPurchases_transformed = np.log(NumDealsPurchases + 1)
    
    NumWebPurchases_transformed=np.sqrt(NumWebPurchases)
    NumCatalogPurchases_transformed=np.sqrt(NumCatalogPurchases)
    # Replace invalid values with 0
    NumStorePurchases = np.where(np.isnan(NumStorePurchases) | (NumStorePurchases < 0), 0, NumStorePurchases)

    # Apply log1p transformation safely
    NumStorePurchases_transformed = np.log1p(NumStorePurchases)    
    NumWebVisitsMonth_transformed=NumWebVisitsMonth

    # Campaign responses
    AcceptedCmp1 = st.selectbox("Accepted Campaign 1?", [0, 1])
    AcceptedCmp2 = st.selectbox("Accepted Campaign 2?", [0, 1])
    AcceptedCmp3 = st.selectbox("Accepted Campaign 3?", [0, 1])
    AcceptedCmp4 = st.selectbox("Accepted Campaign 4?", [0, 1])
    AcceptedCmp5 = st.selectbox("Accepted Campaign 5?", [0, 1])

    # Total campaign
    campaign=AcceptedCmp1+AcceptedCmp2+AcceptedCmp3+AcceptedCmp4+AcceptedCmp5

    Dt_Customer = st.date_input("Customer Enrollment Date", value=datetime(2014, 1, 1))
    Customer_Since_Days = (pd.Timestamp.today().date() - Dt_Customer).days

    Education_value = st.selectbox(
        "Education Level", 
        ["Basic", "2n Cycle", "Graduation", "Master", "PhD"]
    )
    Education_map={"Basic":1, "2n Cycle":2, "Graduation":3, "Master":4, "PhD":5}
    Education_Ordinal=Education_map[Education_value]

    # Initialize all as 0
    Marital_Status_Married = 0
    Marital_Status_Single = 0
    Marital_Status_Together = 0
    Marital_Status_Widow = 0

    # Set 1 for the selected category
    if Marital_Status == "Married":
        Marital_Status_Married = 1
    elif Marital_Status == "Single":
        Marital_Status_Single = 1
    elif Marital_Status == "Together":
        Marital_Status_Together = 1
    elif Marital_Status == "Widow":
        Marital_Status_Widow = 1

    

    #  Create input DataFrame (must match training features)
    input_df = pd.DataFrame([[
        Income, Response, Age, FamilySize, MntWines_transformed, MntFruits_transformed, MntMeatProducts_transformed, MntFishProducts_transformed, MntSweetProducts_transformed, MntGoldProds_transformed, NumDealsPurchases_transformed, NumWebPurchases_transformed, NumCatalogPurchases_transformed, NumStorePurchases_transformed, NumWebVisitsMonth_transformed, Customer_Since_Days, Education_Ordinal, Marital_Status_Married, Marital_Status_Single, Marital_Status_Together, Marital_Status_Widow, campaign

    ]], columns=['Income', 'Response', 'Age', 'FamilySize', 'MntWines_transformed',
       'MntFruits_transformed', 'MntMeatProducts_transformed',
       'MntFishProducts_transformed', 'MntSweetProducts_transformed',
       'MntGoldProds_transformed', 'NumDealsPurchases_transformed',
       'NumWebPurchases_transformed', 'NumCatalogPurchases_transformed',
       'NumStorePurchases_transformed', 'NumWebVisitsMonth_transformed',
       'Customer_Since_Days', 'Education_Ordinal', 'Marital_Status_Married',
       'Marital_Status_Single', 'Marital_Status_Together',
       'Marital_Status_Widow', 'campaign'])

    if st.button("🔍 Predict Cluster"):
        # Preprocess
        scaled_input = scaler.transform(input_df)
        pca_input = pca.transform(scaled_input)
        cluster_label = model.predict(pca_input)[0]

        st.success(f"✅ This customer belongs to **Cluster {cluster_label}**")

        #  simple interpretation
        st.info("""
        🧭 **Cluster Interpretation:**
        - Cluster 0 -> High-Value Loyal Customers
        - Cluster 1 -> Budget-Conscious Shoppers 
        - Cluster 2 -> Family-Oriented Moderate Spenders
        - Cluster 3 -> Inactive or Lost Customers
        - Cluster 4 -> New or Potential High-Spenders
        """)

st.caption("Developed by Siva | Market Customer Segmentation | Streamlit App")

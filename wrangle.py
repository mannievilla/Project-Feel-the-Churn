import os
from env import get_db_url
import pandas as pd

from sklearn.model_selection import train_test_split

#################################Acquire the Data#################################
# this function read the data to create the dataframe, drop unnecessary columns, 
# changes value types and finally adds the dumy columns
def wrangle_telco_data():
    filename = "telco.csv"
    # if filename is found in directory it reads file from saved directory
    if os.path.isfile(filename):
        df = pd.read_csv(filename, index_col=0)
    # else the file is created using this query
    else:
        url = get_db_url('telco_churn')
        df = pd.read_sql('''SELECT * 
                            FROM customers AS c
                            JOIN contract_types AS ct USING (contract_type_id)
                            JOIN internet_service_types AS it USING (internet_service_type_id)
                            JOIN payment_types AS pt USING (payment_type_id)
                            ;''', url)
        # drops the unneeded columns
        df.drop(['payment_type_id', 'internet_service_type_id', 'contract_type_id'], axis=1, inplace=True)                    
        # adds a new column baseline_prediciton takes the most values based on total number of appearance
        df['baseline_prediction'] = 0
        # saves file to this directory
        df.to_csv(filename)
#################################Prep the Data#################################
    telco_df = df
    telco_df['senior_citizen'] = telco_df['senior_citizen'].astype(str)
    # changed the column type to numeric or float64 values
    telco_df['total_charges'] = pd.to_numeric(telco_df['total_charges'], errors='coerce')
    #### train[col].nunique() < 10
    
    col_list = []
    # looping the columns to append to a list if they less than 10 categories so i can dummy those columns
    for col in telco_df: 
        if telco_df[col].nunique() < 10 and col != 'baseline_prediction':
            col_list.append(col)
       
    dummy_df = pd.get_dummies(telco_df[col_list], drop_first=True)
        
    #dummy_df = pd.get_dummies(telco_df[['internet_service_type', 'multiple_lines']], drop_first=True)
    concat_telco = pd.concat([telco_df, dummy_df], axis=1)
    
    return concat_telco

#################################Split the Data#################################

def split_telco_data(df):
    
    train, test = train_test_split(df, 
                random_state=123, 
                test_size=0.20, 
                stratify=df.churn)
    train, validate = train_test_split(train, 
                random_state=123,
                 test_size=.25,
                 stratify= train.churn)
     
    return train, validate, test 
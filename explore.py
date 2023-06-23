# import the necessary libraries to run code
import pandas as pd
import numpy as np
from sys import displayhook
import seaborn as sns
import matplotlib.pyplot as plt
from wrangle import wrangle_telco_data, split_telco_data
from IPython.display import display

# call the data
df = wrangle_telco_data()

train, validate, test = split_telco_data(df)


def dsl_v_churn():


    # total number customers for in train data under 
    none = train[train['internet_service_type']=='None'].churn.value_counts()
    fiber = train[train['internet_service_type']=='Fiber optic'].churn.value_counts()
    dsl = train[train['internet_service_type']=='DSL'].churn.value_counts()

    total_yes = dsl+none+fiber
    total = train['internet_service_type'].value_counts().sum()


    sns.barplot(x='internet_service_type', y='churn_Yes', data=train)
    plt.title('internet_service_type')
    mean = total_yes[1]/total
    plt.axhline(mean, label="Churn Mean", color='red', linestyle='dotted')
    plt.xticks(rotation=45)

    plt.show()


def contract_type_v_churn():
    # pulls out the max value of what contract month has the most churn per customer and contract type
    sum_month_to_month = train[train['contract_type']=='Month-to-month'].groupby('tenure')\
    .churn_Yes.sum().sort_values(ascending=False)
    sum_month_to_month = pd.DataFrame(sum_month_to_month)
    sum_month_to_month['position'] = np.arange(1, len(sum_month_to_month) + 1)  # Add a new column for the index position
  

    # pulls out the max value of what contract month has the most churn per customer and contract type
    sum_one_year = train[train['contract_type']=='One year'].groupby('tenure')\
    .churn_Yes.sum().sort_values(ascending=False)
    sum_one_year = pd.DataFrame(sum_one_year)
    sum_one_year['position'] = np.arange(1, len(sum_one_year) + 1)  # Add a new column for the index position


    # pulls out the max value of what contract month has the most churn per customer and contract type    
    sum_two_year = train[train['contract_type']=='Two year'].groupby('tenure')\
    .churn_Yes.sum().sort_values(ascending=False)
    sum_two_year = pd.DataFrame(sum_two_year)
    sum_two_year['position'] = np.arange(1, len(sum_two_year) + 1)  # Add a new column for the index position

    
    # graphs the number of customers per contract type and how many have churned or not
    sns.countplot(x='contract_type', hue='churn', data=train)
    plt.title('Contract Types and Customer Churn')

    plt.xticks(rotation=45)

    plt.show()
    
    # Top month in for month to month contracts
    print('Top churned from Month to Month Contraacts')
    display(sum_month_to_month.head(1) )
    
    # Top month in for month to month contracts
    print('Top churned from One Year Contraacts')    
    display(sum_one_year.head(1)) 
    
    # Top month in for month to month contracts
    print('Top churned from Two Year Contraacts')    
    display(sum_two_year.head(1) )
    
    
def payment_v_churn():
    
    sns.countplot(x='payment_type', hue='churn', data=train)
    plt.title('Contract Types and Customer Churn')
    
    plt.xticks(rotation=45)
    
    plt.show()        
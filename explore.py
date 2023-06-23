# import the necessary libraries to run code
import pandas as pd
import numpy as np
from sys import displayhook
import seaborn as sns
import matplotlib.pyplot as plt
from wrangle import wrangle_telco_data, split_telco_data
from IPython.display import display
import scipy.stats as stats

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


def monthly_v_churn():
    # creates the dataframes that I need to generate the vlaues 
    # from the monthly charges column where churn == 0 or 1
    train_no = train[train['churn_Yes']==0]
    train_yes = train[train['churn_Yes']==1]

    # Create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # setting the data and columns, adding labels, median line, 
    # adding text to to the line and legend on ax1(subplot)
    sns.histplot(x='monthly_charges', hue='churn', data=train_yes, ax=ax1)
    ax1.set_title('Monthly Charges and Customer Churn')
    ax1.set_ylabel('Customer Count')
    median1 = train_yes['monthly_charges'].median()
    ax1.axvline(median1, label="Monthly Charges Median", color='red', linestyle='dotted')
    ax1.annotate(text=round(train_yes['monthly_charges'].median(),2), xy=(80,150)) 
    ax1.legend()

    sns.histplot(x='monthly_charges', hue='churn', data=train_no, ax=ax2, palette=['orange'])
    #needs palettte=['orange'] to add orange color on No churn

    # Setting title, label, median line, adding text to the line and legend on ax2(subplot)
    ax2.set_title('Monthly Charges and Customer Churn') # neeed ax2.set_title to add title on weach block
    ax2.set_ylabel('Customer Count')
    median2 = train_no['monthly_charges'].median()
    ax2.axvline(median2, label="Monthly Charges Median", color='red', linestyle='dotted')
    ax2.annotate(text=round(train_no['monthly_charges'].median(),2), xy=(67,500))
    ax2.legend()

    plt.show()    

    train_70 = train[train['monthly_charges']>=70]
    train_70_yes = train_70[train_70['churn_Yes']==1]
    train_70_no = train_70[train_70['churn_Yes']==0]

    # Create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    sns.histplot(x='monthly_charges', hue='churn', data=train_70_yes, ax=ax1)
    ax1.set_title('Monthly Charges and Customer Churn')
    ax1.set_ylabel('Customer Count')

    plt.xticks(rotation=45)


    sns.histplot(x='monthly_charges', hue='churn', data=train_70_no, ax=ax2, palette=['orange'])
    #needs palettte=['orange'] to add orange color on No churn
    ax2.set_title('Monthly Charges and Customer Churn') # neeed ax2.set_title to add title on weach block
    ax2.set_ylabel('Customer Count')

    plt.xticks(rotation=45)


    plt.show()

    

def hyp_month_v_churn():
    # setting alpha for 95% accuracy
    alpha = 0.05

    # creating dataframes to what customers churn
    trn_mon_churn_yes = train[train['churn_Yes']==1].monthly_charges
    trn_mon_churn_no = train[train['churn_Yes']==0].monthly_charges

    # plotting to see the distribution
    plt.figure(figsize=(9,6))
    plt.hist([trn_mon_churn_yes , trn_mon_churn_no], label=["Churn", "No Churn"])
    plt.legend(loc="upper right")



def t_test_monthly_churn():
    # creating dataframes to what customers churn
    trn_mon_churn_yes = train[train['churn_Yes']==1].monthly_charges
    trn_mon_churn_no = train[train['churn_Yes']==0].monthly_charges 
    # running the stats test #NOTE: variance ws not equal so, equal_var=False
    t, p = stats.ttest_ind(trn_mon_churn_yes, trn_mon_churn_no, equal_var=False)
    # return t and p
    return t, p

def chi2_payment_churn():
    # creating the observeed dataframe comparing the payment type and churn
    observed = pd.crosstab(train['payment_type'], train['churn_Yes'])
    # running the chi^2 test
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    # return the chi^2 and p
    return chi2, p

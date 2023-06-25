import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from wrangle import split_telco_data, wrangle_telco_data
# create X & y version of train/validate/test
# where X contains the features we want to use and y is a series with just the target variable

def model_prep_data():
    df = wrangle_telco_data()
    # splits the data
    train, validate, test = split_telco_data(df)

    # creates the features (X_train) I will be running through the model
    # creates the target for train data
    X_train = train.drop(columns=['customer_id', 'gender', 'senior_citizen', 'partner', 'dependents',
        'tenure', 'phone_service', 'multiple_lines', 'online_security',
        'online_backup', 'device_protection', 'tech_support', 'streaming_tv',
        'streaming_movies', 'paperless_billing',
        'total_charges', 'churn', 'contract_type', 'internet_service_type',
        'payment_type', 'baseline_prediction', 'gender_Male',
        'senior_citizen_1', 'partner_Yes', 'dependents_Yes',
        'phone_service_Yes', 'multiple_lines_No phone service',
        'multiple_lines_Yes', 'online_security_No internet service',
        'online_security_Yes', 'online_backup_No internet service',
        'online_backup_Yes', 'device_protection_No internet service',
        'device_protection_Yes', 'tech_support_No internet service',
        'tech_support_Yes', 'streaming_tv_No internet service',
        'streaming_tv_Yes', 'streaming_movies_No internet service',
        'streaming_movies_Yes', 'paperless_billing_Yes', 'churn_Yes'])
    y_train = train.churn_Yes

    # Creates the feature (X_validate) runnig through the model
    # creates the target for validate data
    X_validate = validate.drop(columns=['customer_id', 'gender', 'senior_citizen', 'partner', 'dependents',
        'tenure', 'phone_service', 'multiple_lines', 'online_security',
        'online_backup', 'device_protection', 'tech_support', 'streaming_tv',
        'streaming_movies', 'paperless_billing',
        'total_charges', 'churn', 'contract_type', 'internet_service_type',
        'payment_type', 'baseline_prediction', 'gender_Male',
        'senior_citizen_1', 'partner_Yes', 'dependents_Yes',
        'phone_service_Yes', 'multiple_lines_No phone service',
        'multiple_lines_Yes', 'online_security_No internet service',
        'online_security_Yes', 'online_backup_No internet service',
        'online_backup_Yes', 'device_protection_No internet service',
        'device_protection_Yes', 'tech_support_No internet service',
        'tech_support_Yes', 'streaming_tv_No internet service',
        'streaming_tv_Yes', 'streaming_movies_No internet service',
        'streaming_movies_Yes', 'paperless_billing_Yes', 'churn_Yes'])
    y_validate = validate.churn_Yes

    # Creates the feature (X_test) runnig through the model
    # creates the target for test data
    X_test = test.drop(columns=['customer_id', 'gender', 'senior_citizen', 'partner', 'dependents',
        'tenure', 'phone_service', 'multiple_lines', 'online_security',
        'online_backup', 'device_protection', 'tech_support', 'streaming_tv',
        'streaming_movies', 'paperless_billing',
        'total_charges', 'churn', 'contract_type', 'internet_service_type',
        'payment_type', 'baseline_prediction', 'gender_Male',
        'senior_citizen_1', 'partner_Yes', 'dependents_Yes',
        'phone_service_Yes', 'multiple_lines_No phone service',
        'multiple_lines_Yes', 'online_security_No internet service',
        'online_security_Yes', 'online_backup_No internet service',
        'online_backup_Yes', 'device_protection_No internet service',
        'device_protection_Yes', 'tech_support_No internet service',
        'tech_support_Yes', 'streaming_tv_No internet service',
        'streaming_tv_Yes', 'streaming_movies_No internet service',
        'streaming_movies_Yes', 'paperless_billing_Yes', 'churn_Yes'])
    y_test = test.churn_Yes

    return X_train, y_train, X_validate, y_validate, X_test, y_test

# creates the datfram
df = wrangle_telco_data()
# splits the data
train, validate, test = split_telco_data(df)

# creates the features (X_train) I will be running through the model
# creates the target for train data
X_train = train.drop(columns=['customer_id', 'gender', 'senior_citizen', 'partner', 'dependents',
       'tenure', 'phone_service', 'multiple_lines', 'online_security',
       'online_backup', 'device_protection', 'tech_support', 'streaming_tv',
       'streaming_movies', 'paperless_billing',
       'total_charges', 'churn', 'contract_type', 'internet_service_type',
       'payment_type', 'baseline_prediction', 'gender_Male',
       'senior_citizen_1', 'partner_Yes', 'dependents_Yes',
       'phone_service_Yes', 'multiple_lines_No phone service',
       'multiple_lines_Yes', 'online_security_No internet service',
       'online_security_Yes', 'online_backup_No internet service',
       'online_backup_Yes', 'device_protection_No internet service',
       'device_protection_Yes', 'tech_support_No internet service',
       'tech_support_Yes', 'streaming_tv_No internet service',
       'streaming_tv_Yes', 'streaming_movies_No internet service',
       'streaming_movies_Yes', 'paperless_billing_Yes', 'churn_Yes'])
y_train = train.churn_Yes

# Creates the feature (X_validate) runnig through the model
# creates the target for validate data
X_validate = validate.drop(columns=['customer_id', 'gender', 'senior_citizen', 'partner', 'dependents',
       'tenure', 'phone_service', 'multiple_lines', 'online_security',
       'online_backup', 'device_protection', 'tech_support', 'streaming_tv',
       'streaming_movies', 'paperless_billing',
       'total_charges', 'churn', 'contract_type', 'internet_service_type',
       'payment_type', 'baseline_prediction', 'gender_Male',
       'senior_citizen_1', 'partner_Yes', 'dependents_Yes',
       'phone_service_Yes', 'multiple_lines_No phone service',
       'multiple_lines_Yes', 'online_security_No internet service',
       'online_security_Yes', 'online_backup_No internet service',
       'online_backup_Yes', 'device_protection_No internet service',
       'device_protection_Yes', 'tech_support_No internet service',
       'tech_support_Yes', 'streaming_tv_No internet service',
       'streaming_tv_Yes', 'streaming_movies_No internet service',
       'streaming_movies_Yes', 'paperless_billing_Yes', 'churn_Yes'])
y_validate = validate.churn_Yes

# Creates the feature (X_test) runnig through the model
# creates the target for test data
X_test = test.drop(columns=['customer_id', 'gender', 'senior_citizen', 'partner', 'dependents',
       'tenure', 'phone_service', 'multiple_lines', 'online_security',
       'online_backup', 'device_protection', 'tech_support', 'streaming_tv',
       'streaming_movies', 'paperless_billing',
       'total_charges', 'churn', 'contract_type', 'internet_service_type',
       'payment_type', 'baseline_prediction', 'gender_Male',
       'senior_citizen_1', 'partner_Yes', 'dependents_Yes',
       'phone_service_Yes', 'multiple_lines_No phone service',
       'multiple_lines_Yes', 'online_security_No internet service',
       'online_security_Yes', 'online_backup_No internet service',
       'online_backup_Yes', 'device_protection_No internet service',
       'device_protection_Yes', 'tech_support_No internet service',
       'tech_support_Yes', 'streaming_tv_No internet service',
       'streaming_tv_Yes', 'streaming_movies_No internet service',
       'streaming_movies_Yes', 'paperless_billing_Yes', 'churn_Yes'])
y_test = test.churn_Yes





#############################Decision Tree#############################


def model_decision_tree(X, X_val, y, y_val):
    
    clf = DecisionTreeClassifier(max_depth=4, random_state=123)

    trn_score = clf.score(X, y)
    clf.fit(X, y)
    val_score = clf.score(X_val, y_val)
    print(f"Accuracy of Decision Tree on train data is: {trn_score}")
    print(f"Accuracy of Decision Tree on validate data is: {val_score}")


def decision_tree():
    i = 1
    # storing my values to use them to create dictiojnaries containg 

    # checking the baseline accuracy
    baseline_accuracy = (df.churn_Yes == df.baseline_prediction).mean()

    while i < 11:
        
        clf = DecisionTreeClassifier(max_depth=i, random_state=123)
        clf_fit = clf.fit(X_train, y_train)
        
        # Compute score
        trn_score = clf.score(X_train, y_train)
        val_score = clf.score(X_validate, y_validate)
        y_train_pred = clf_fit.predict(X_train)
        trn_report = classification_report(y_train, y_train_pred)
        y_val_pred = clf.predict(X_validate)
        val_report = classification_report(y_validate, y_val_pred)


        print(f'Training Dataset Model with Max Depth of {i},')

        # Print the model's accuracy and other information
        print(f"Model's Accuracy: {trn_score}")
        print(f"Difference between Model and Basleine Accuracy: {trn_score - baseline_accuracy}")
        print('Train Classification Report')
        print(trn_report)
        print()
        print('           VS             ')
        print()
        print(f'Validation Dataset Model with Max Depth of {i},')

        print(f"Model's Accuracy: {val_score}")
        print(f"Difference between Model and Basleine Accuracy: {val_score - baseline_accuracy}")
        print('Validate Classification Report')
        print(val_report)
        print()
        print(f'Difference bewtween Training and Validate:{trn_score-val_score}')
        #print(val_report)
        print()
        print('----------------------------------------------------')
        print()

        # Increment 'i' and 'j' for the next iteration
        i += 1




def dictionary_values():
    train_depth = []
    train_score_list = [] 
    val_score_list = []
    trn_report_list = []
    val_report_list = []
    diff_list = []

    for i in range(1, 11):
        clf = DecisionTreeClassifier(max_depth=i, random_state=123)
        clf_fit = clf.fit(X_train, y_train)
    
        # Compute score
        trn_score = clf.score(X_train, y_train)
        val_score = clf.score(X_validate, y_validate)
        y_train_pred = clf_fit.predict(X_train)
        trn_report = classification_report(y_train, y_train_pred)
        y_val_pred = clf.predict(X_validate)
        val_report = classification_report(y_validate, y_val_pred)
        val_report_list.append(val_report)
        train_score_list.append(trn_score)
        val_score_list.append(val_score)
        train_depth.append(i)

        diff = trn_score - val_score
        diff_list.append(diff)

    return train_depth, train_score_list, val_score_list, diff_list






def analysis_tree():

    train_depth,train_score_list, val_score_list, diff_list = dictionary_values()

    train_dict = {x: y for x, y in zip(train_depth, train_score_list)}
    val_dict = {x: y for x, y in zip(train_depth, val_score_list)}
    diff_dict = {x: y for x, y in zip(train_depth, diff_list)}

    return train_dict, val_dict, diff_dict
    
    


def print_tree_reprt(): 

    train_dict, val_dict, diff_dict = analysis_tree()

    # model number to iterate
    model_number = 1   
    print('*************FINAL ANALYSIS*************')
    
    for key, value in diff_dict.items():
        if value == min(value for value in diff_dict.values() if value >= 0.009):
            print()
            print(f'Top Model #{model_number}')
            
            print()
            print(f'''Top performing Training Model:
            
    Max Depth:{key}
    Accuracy:{max(train_dict.values())}
            
    Top performing Validation Model:

    Max Depth:{key}
    Accuracy:{max(val_dict.values())}
    Difference:{value}''')
            model_number+=1
            print('---------------------------------')
    
       
     



def difference_graph():
    train_dict, val_dict, diff_dict = analysis_tree()
    # looping the list of differnces in train vs validate
    key_values = []
    for key, value in diff_dict.items():
        key_values.append(key)
    # pulling the values for each iteration    
    train_values = train_dict.values()
    validate_values = val_dict.values()
    diff_values = diff_dict.values()
    # using the values to create a dataframe
    diff_df = pd.DataFrame({'depth':key_values, 'train_acc':train_values, 'validate_acc':validate_values,\
                            'difference':diff_values})

    # graphing the dataframe
    diff_df.set_index('depth').plot(figsize = (16,9))
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0,10,1))
    plt.grid()



############################# K N N #############################



def model_knn(X, X_val, y, y_val):
    
    knn = KNeighborsClassifier(n_neighbors=15)

    trn_score = knn.score(X, y)
    knn.fit(X, y)
    val_score = knn.score(X_val, y_val)
    print(f"Accuracy of KNN on train data is: {trn_score}")
    print(f"Accuracy of KNN on validate data is: {val_score}")

def knn_model():
    i = 1
    # storing my values to use them to create dictiojnaries containg 

    # checking the baseline accuracy
    baseline_accuracy = (df.churn_Yes == df.baseline_prediction).mean()

    while i < 20:
        
        knn = KNeighborsClassifier(n_neighbors=i)
        knn_fit = knn.fit(X_train, y_train)
        
        # Compute score
        trn_score = knn.score(X_train, y_train)
        val_score = knn.score(X_validate, y_validate)
        y_train_pred = knn_fit.predict(X_train)
        trn_report = classification_report(y_train, y_train_pred)
        y_val_pred = knn.predict(X_validate)
        val_report = classification_report(y_validate, y_val_pred)


        print(f'Training Dataset Model with Max Depth of {i},')

        # Print the model's accuracy and other information
        print(f"Model's Accuracy: {trn_score}")
        print(f"Difference between Model and Basleine Accuracy: {trn_score - baseline_accuracy}")
        print('Train Classification Report')
        print(trn_report)
        print()
        print('           VS             ')
        print()
        print(f'Validation Dataset Model with Max Depth of {i},')

        print(f"Model's Accuracy: {val_score}")
        print(f"Difference between Model and Basleine Accuracy: {val_score - baseline_accuracy}")
        print('Validate Classification Report')
        print(val_report)
        print()
        print(f'Difference bewtween Training and Validate:{trn_score-val_score}')
        #print(val_report)
        print()
        print('----------------------------------------------------')
        print()

        # Increment 'i' and 'j' for the next iteration
        i += 1




def knn_dictionary_values():
    train_depth = []
    train_score_list = [] 
    val_score_list = []
    trn_report_list = []
    val_report_list = []
    diff_list = []

    for i in range(1, 21):
        knn= KNeighborsClassifier(n_neighbors=i)
        knn_fit = knn.fit(X_train, y_train)
    
        # Compute score
        trn_score = knn.score(X_train, y_train)
        val_score = knn.score(X_validate, y_validate)
        y_train_pred = knn_fit.predict(X_train)
        trn_report = classification_report(y_train, y_train_pred)
        y_val_pred = knn.predict(X_validate)
        val_report = classification_report(y_validate, y_val_pred)
        val_report_list.append(val_report)
        train_score_list.append(trn_score)
        val_score_list.append(val_score)
        train_depth.append(i)

        diff = trn_score - val_score
        diff_list.append(diff)

    return train_depth, train_score_list, val_score_list, diff_list




def knn_analysis():

    train_depth,train_score_list, val_score_list, diff_list = knn_dictionary_values()

    train_dict = {x: y for x, y in zip(train_depth, train_score_list)}
    val_dict = {x: y for x, y in zip(train_depth, val_score_list)}
    diff_dict = {x: y for x, y in zip(train_depth, diff_list)}

    return train_dict, val_dict, diff_dict
    
    


def print_knn_reprt(): 

    train_dict, val_dict, diff_dict = knn_analysis()

    # model number to iterate
    model_number = 1   
    print('*************FINAL ANALYSIS*************')
    
    for key, value in diff_dict.items():
        if value == min(diff_dict.values()):
            print()
            print(f'Top Model #{model_number}')
            
            print()
            print(f'''Top performing Training Model:
            
    Max Depth:{key}
    Accuracy:{train_dict[key]}
            
    Top performing Validation Model:

    Max Depth:{key}
    Accuracy:{val_dict[key]}
    Difference:{value}''')
            model_number+=1
            print('---------------------------------')
    
       
     



def knn_difference_graph():
    train_dict, val_dict, diff_dict = knn_analysis()
    # looping the list of differnces in train vs validate
    key_values = []
    for key, value in diff_dict.items():
        key_values.append(key)
    # pulling the values for each iteration    
    train_values = train_dict.values()
    validate_values = val_dict.values()
    diff_values = diff_dict.values()
    # using the values to create a dataframe
    diff_df = pd.DataFrame({'depth':key_values, 'train_acc':train_values, 'validate_acc':validate_values,\
                            'difference':diff_values})

    # graphing the dataframe
    diff_df.set_index('depth').plot(figsize = (16,9))
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0,20,1))
    plt.grid()




############################# Log Regression #############################


def model_log(X, X_val, y, y_val):
    
    log = LogisticRegression(C=10, 
                            class_weight={0:1, 1:99}, 
                            random_state=123, 
                            intercept_scaling=1, 
                            solver='lbfgs')

    trn_score = log.score(X, y)
    log.fit(X, y)
    val_score = log.score(X_val, y_val)
    print(f"Accuracy of Log Regression on train data is: {trn_score}")
    print(f"Accuracy of Log Regression on validate data is: {val_score}")




def log_model():
    c = [.001, .01, .1, 1, 10, 100, 1000]
    # storing my values to use them to create dictiojnaries containg 

    # checking the baseline accuracy
    baseline_accuracy = (df.churn_Yes == df.baseline_prediction).mean()

    for c in c:
        
        log = LogisticRegression(C=c, 
                            class_weight={0:1, 1:99}, 
                            random_state=123, 
                            intercept_scaling=1, 
                            solver='lbfgs')
        log_fit = log.fit(X_train, y_train)
        
        # Compute score
        trn_score = log.score(X_train, y_train)
        val_score = log.score(X_validate, y_validate)
        y_train_pred = log_fit.predict(X_train)
        trn_report = classification_report(y_train, y_train_pred)
        y_val_pred = log.predict(X_validate)
        val_report = classification_report(y_validate, y_val_pred)


        print(f'Training Dataset Model Coefficient of {c},')

        # Print the model's accuracy and other information
        print(f"Model's Accuracy: {trn_score}")
        print(f"Difference between Model and Basleine Accuracy: {trn_score - baseline_accuracy}")
        print('Train Classification Report')
        print(trn_report)
        print()
        print('           VS             ')
        print()
        print(f'Validation Dataset Model Coefficient of {c},')

        print(f"Model's Accuracy: {val_score}")
        print(f"Difference between Model and Basleine Accuracy: {val_score - baseline_accuracy}")
        print('Validate Classification Report')
        print(val_report)
        print()
        print(f'Difference bewtween Training and Validate:{trn_score-val_score}')
        #print(val_report)
        print()
        print('----------------------------------------------------')
        print()

        # Increment 'i' and 'j' for the next iteration
    




def log_dictionary_values():

    c = [.001, .01, .1, 1, 10, 100, 1000]

    train_depth = []
    train_score_list = [] 
    val_score_list = []
    trn_report_list = []
    val_report_list = []
    diff_list = []

    for c in c:

        log = LogisticRegression(C=c, 
                            class_weight={0:1, 1:99}, 
                            random_state=123, 
                            intercept_scaling=1, 
                            solver='lbfgs')
        log_fit = log.fit(X_train, y_train)
    
        # Compute score
        trn_score = log.score(X_train, y_train)
        val_score = log.score(X_validate, y_validate)
        y_train_pred = log_fit.predict(X_train)
        trn_report = classification_report(y_train, y_train_pred)
        y_val_pred = log.predict(X_validate)
        val_report = classification_report(y_validate, y_val_pred)
        val_report_list.append(val_report)
        train_score_list.append(trn_score)
        val_score_list.append(val_score)
        train_depth.append(c)

        diff = trn_score - val_score
        diff_list.append(diff)

    return train_depth, train_score_list, val_score_list, diff_list




def log_analysis():

    train_depth,train_score_list, val_score_list, diff_list = log_dictionary_values()

    train_dict = {x: y for x, y in zip(train_depth, train_score_list)}
    val_dict = {x: y for x, y in zip(train_depth, val_score_list)}
    diff_dict = {x: y for x, y in zip(train_depth, diff_list)}

    return train_dict, val_dict, diff_dict
    
    


def print_log_reprt(): 

    train_dict, val_dict, diff_dict = log_analysis()

    # model number to iterate
    model_number = 1   
    print('*************FINAL ANALYSIS*************')
    
    for key, value in diff_dict.items():
        if value == min(diff_dict.values()):
            print()
            print(f'Top Model #{model_number}')
            
            print()
            print(f'''Top performing Training Model:
            
    Model Coefficient:{key}
    Accuracy:{train_dict[key]}
            
    Top performing Validation Model:

    Max Depth:{key}
    Accuracy:{val_dict[key]}
    Difference:{value}''')
            model_number+=1
            print('---------------------------------')



def log_difference_graph():
    train_dict, val_dict, diff_dict = log_analysis()
    # looping the list of differnces in train vs validate
    key_values = []
    for key, value in diff_dict.items():
        key_values.append(key)
    # pulling the values for each iteration    
    train_values = train_dict.values()
    validate_values = val_dict.values()
    diff_values = diff_dict.values()
    # using the values to create a dataframe
    diff_df = pd.DataFrame({'depth':key_values, 'train_acc':train_values, 'validate_acc':validate_values,\
                            'difference':diff_values})

    # graphing the dataframe
    diff_df.set_index('depth').plot(figsize = (16,9))
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0,7,1))
    plt.grid()





############################# Random Forest #############################





def forest_model():

    i = 10
    j = 1
    baseline_accuracy = (df.churn_Yes == df.baseline_prediction).mean()
    # storing my values to use them to create dictiojnaries containg 
    # all the collected data for head to head comaprison


    while j < 11:
            
            rf = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=j,
                            n_estimators=100,
                            max_depth=i, 
                            random_state=123)
            rf_fit = rf.fit(X_train, y_train)
            # Compute score
            trn_score = rf.score(X_train, y_train)
            val_score = rf.score(X_validate, y_validate)
            y_train_pred = rf_fit.predict(X_train)
            trn_report = classification_report(y_train, y_train_pred)
            y_val_pred = rf.predict(X_validate)
            val_report = classification_report(y_validate, y_val_pred)
    #         trn_report_list.append(trn_report)

            
            print(f'Training Dataset Model with Max Depth of {i},')
            print(f'And Leaf Level of {j}:')
            
            # Print the model's accuracy and other information
            print(f"Model's Accuracy: {trn_score}")
            print(f"Difference between Model and Basleine Accuracy: {trn_score - baseline_accuracy}")
            print('Train Classification Report')
            print(trn_report)
            print()
            print('           VS             ')
            print()
            print(f'Validation Dataset Model with Max Depth of {i},')
            print(f'And Leaf Level of {j}:')
            print(f"Model's Accuracy: {val_score}")
            print(f"Difference between Model and Basleine Accuracy: {val_score - baseline_accuracy}")
            print('Validate Classification Report')
            print(val_report)
            print()
            print(f'Difference bewtween Training and Validate:{trn_score-val_score}')
            #print(val_report)
            print()
            print('----------------------------------------------------')
            print()

            # Increment 'i' and 'j' for the next iteration
            i -= 1
            j += 1

def forest_dictionary_values():
            

    i = 10
    j = 1

    # storing my values to use them to create dictiojnaries containg 
    # all the collected data for head to head comaprison
    train_depth = []
    leaf_lvl = []
    train_score_list = [] 
    val_score_list = []
    trn_report_list = []
    val_report_list = []
    diff_list = []

    while j < 11:
            rf = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=j,
                            n_estimators=100,
                            max_depth=i, 
                            random_state=123)
            rf_fit = rf.fit(X_train, y_train)            
            
            # Compute score
            trn_score = rf.score(X_train, y_train)
            val_score = rf.score(X_validate, y_validate)
            y_train_pred = rf_fit.predict(X_train)
            trn_report = classification_report(y_train, y_train_pred)
            y_val_pred = rf.predict(X_validate)
            val_report = classification_report(y_validate, y_val_pred)
            val_report_list.append(val_report)
            train_score_list.append(trn_score)
            val_score_list.append(val_score)
            train_depth.append(i)
            leaf_lvl.append(j)

            diff = trn_score-val_score
            diff_list.append(diff)

            i -= 1
            j += 1

    return train_depth, train_score_list, val_score_list, diff_list, leaf_lvl

def forest_analysis():

    train_depth, train_score_list, val_score_list, diff_list, leaf_lvl = forest_dictionary_values()

    # create dictionaries for my data collection
    train_dict = {(x, y): z for x, y, z in zip(train_depth, leaf_lvl, train_score_list)}
    val_dict = {(x, y): z for x, y, z in zip(train_depth, leaf_lvl, val_score_list)}
    diff_dict = {(x, y): z for x, y, z in zip(train_depth, leaf_lvl, diff_list)}

    return train_dict, val_dict, diff_dict


def print_forest_report(): 

    train_dict, val_dict, diff_dict = forest_analysis()

    # model number to iterate
    # model_number = 1
    
    model_number = 1
    for key, value in diff_dict.items():
        if value == min(value for value in diff_dict.values() if value >= 0.01):
            print()
            print(f'Top Model #{model_number}')
            print('*************FINAL ANALYSIS*************')
            print()
            print(f'''Top performing Training Model:
            Max Depth:{key[0]}
            Leaf Level:{key[1]}
            Accuracy:{train_dict[(key[0],key[1])]}
            
    Top performing Validation Model:

            Max Depth:{key[0]}
            Leaf Level:{key[1]}
            Accuracy:{val_dict[(key[0],key[1])]}
            Difference:{diff_dict[(key[0],key[1])]}''')
            model_number+=1



def forest_difference_graph():

    train_dict, val_dict, diff_dict = forest_analysis()
    # looping the list of differnces in train vs validate
    key_values = []
    for key, value in diff_dict.items():
        key=key[0]
        key_values.append(key)
    # pulling the values for each iteration    
    train_values = train_dict.values()
    validate_values = val_dict.values()
    diff_values = diff_dict.values()
    # using the values to create a dataframe
    df = pd.DataFrame({'depth':key_values, 'train_acc':train_values, 'validate_acc':validate_values, 'difference':diff_values})
    # graphing the dataframe
    df.set_index('depth').plot(figsize = (16,9))
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(0,10,1))
    plt.grid()
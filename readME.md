# Feel the Churn
 
# Project Description
 
A business is only as valuable as it's current and new clients or cumstomers. But what keeps them happy? Price? Service? Product? Or other? In this project we decided to do just that. We will breaking down the data and seeing which element or elements are held in most regard to cutomer loyalty and retention. 
 
# Project Goal
 
* Discover drivers that cause customer to churn
* Use drivers to develop a machine learning model to classify the customer as churn or not. 
* An churning is defined as leaving the company. 
* This information could be used to further our understanding of customer loyalty and empathy.
 
# Initial Thoughts
 
My initial hypothesis is that drivers of churn will be about price and what service the customer recieves for that price.
 
# The Plan
 
* Aquire data from database
 
* Prepare data
   * Create Engineered columns from existing data
       * baseline
       * rating_difference
       * game_rating
       * lower_rated_white
       * time_control_group
 
* Explore data in search of drivers for churn
   * Answer the following initial questions
       * Are customers with DSL more or less likely to churn?
       * What month are customers most likely to churn and does that depend on their contract type?
       * Is there a service that is associated with more churn than expected?
       * Do customers who churn have a higher average monthly spend than those who don't?
      
* Develop a Model to predict if a customer will churn
   * Use drivers identified in explore to build predictive models of different types
   * Evaluate models on train and validate data
   * Select the best model based on highest accuracy
   * Evaluate the best model on test data
 
* Draw conclusions
 
# Data Dictionary

| Feature | Definition |
|:--------|:-----------|
|customer_id| The customer unique ID number|
|gender| Male or Female,
|senior_citizen| 0 or 1, The customer is a senior citizen|
|partner| Yes or No, Customer is a partner with company|
|dependents| Yes or No, Customer has dependents|
|tenure| The time in months how long the customer was or is with the company|
|phone_service| Yes or No, Custoner has phone serivce with pan|
|multiple_lines| Yes, No or No Phone Service; Customer has multiple lines in plan|
|online_security| Yes, No or No Internet Service; Customer has online security with plan|
|online_backup| Yes, No or No Internet Service; |
|online_backup| Yes, No or No Internet Service; 
|device_protection| Yes, No or No Internet Service; 
|tech_support| Yes, No or No Internet Service; 
|streaming_tv| Yes, No or No Internet Service; 
|streaming_movies| Yes, No or No Internet Service; 
|paperless_billing| 
|monthly_charges|
|total_charges|
|churn|
|contract_type|
|internet_service_type|
|payment_type|
|additional features|Encoded values for categorical data

# Steps to Reproduce
1) Clone this repo.
2) Acquire the data from 
3) Put the data in the file containing the cloned repo.
4) Run notebook.
 
# Takeaways and Conclusions
* Customers with Fiber Optics have more churn than DSL.
* Encouraging customers to be on automatic payment plan will seems to reduce churn.
* 643 manual check writers churned which is a 45% churn rate for all payment types.
* When the monthly charges reached approximate \$70 the churn rate rised.
* The median monthly payment for customers who churns is \$79.70
* Customers who do not churn makeup 73% of the data
* The final model failed to significantly outperform the baseline.
* Possible reasons include:
    “payment_type” and “contract_type” may not have had meaningful relation to who will churn.
    Since monthly charges" seems to be a larger contributor to churn, adding more of the services to see which service may be contributing to churn. 
 
# Recommendations
* This may be simple enough but have a column for reason for caneling service. Helpful to pinpoint issues and improve service.


# Next Steps
* Explore the relation of Fiber Optics to churn. Services like tech support or streaming services could also be explored.

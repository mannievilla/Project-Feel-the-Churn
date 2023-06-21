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
|Rated| True or False, The game's result is reflected in each player's rating|
|Winning Pieces| The color of pieces the winning player was moving|
|White Rating| Rating of the player moving the white pieces using the Glicko-2 rating method for games played on Lichess|
|Black Rating| Rating of the player moving the white pieces using the Glicko-2 rating method for games played on Lichess|
|Rating Difference| The difference in rating between the players in the game|
|Game Rating| The average rating of the two players in the game|
|Lower Rated White| True or False, The lower rated player is moving the white pieces|
|Opening Name| The name of the opening played in the game|
|Time Control Group| The amount of time allotted to each player to make their moves, **Standard** (60 min or more), **Rapid** (30 - 15 min), **Blitz** (5 - 3 min), or **Bullet** (2 or less), **Other** (any other time limit)|
|Upset (Target)| True or False, The lower rated player won the game|
|Additional Features|Encoded and values for categorical data and scaled versions continuous data|
 
# Steps to Reproduce
1) Clone this repo.
2) Acquire the data from 
3) Put the data in the file containing the cloned repo.
4) Run notebook.
 
# Takeaways and Conclusions
* Upsets occur in 1/3 of games
* In games where the lower rated player moves first there is a 4% greater chance of an upset
* Games that are rated have a 3% higher chance of an upset
* Games with a "quick" time control (30 min or less) have about a 1 in 3 chance of upset
* Games with a "slow" time control (60 min or more) have about a 1 in 5 chance of upset
* The mean rating of players in a game is not a driver of upsets
* The difference in player rating is a driver of upsets
* A player's choice of opening is a driver of upsets, however its influence is complicated and I would need more time to discover what role it plays
 
# Recommendations
* To increase the skill intensity of a game add to the length of time players are able to consider their moves
* Based on the data longer time controls make it less likely for a less skilled player to beat a more skilled player
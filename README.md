# Softbank_USD-JPY
# Prepare train and test
Extracting Information from text features and combine It with training and test data
Text feature of a particular date ‘id’ is passed from the lstm model on one unit and return sequences, return the hidden state output for each input time step is stored. This hidden is of shape (1,300) for a single ‘id’. Similarly, for all 5236 id’s of 241 chunk csv files, hidden state is stored and finally concat the hidden states giving shape (5236,300) called final_lstm.csv
Keeping only the id’s which are present training data, and then arranged them in sequence and then combine them by concat both datasets creating the shape of (4176,662). Similarly done for the test data.
Train and test dataset were ready to process further.

# Data preprocessing of train and test
Missing values are handled and training data was divided into target and X. Training data is split and cross-validate data was created.
# Prepare the model and get the prediction
Different ensemble models were tried including XGBRegressor, GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor, lightgbm are used. Voting regressor was also used to get a better prediction.
Bayesian optimization for automated hyperparameter tuning was used by using hyperopt library.

## Best Model = ExtraTreesRegressor(n_estimators=500,random_state=1234)

final_lstm: set of hidden state of text feature of all id’s
traineco_lstm: combined data of final_lstm and given train
testeco_lstm: combined data of final_lstm and test data
extra_tree_500_bestmodel: best submission model acc-> 0.98098

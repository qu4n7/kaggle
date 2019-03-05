import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)

home_data.columns

y = home_data.SalePrice
print(y.shape)
print(y[:5])

feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 
                 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[feature_columns]

X.describe()

X.head()

X.shape

# specify model
iowa_model = DecisionTreeRegressor(random_state = 1)
# fit model
iowa_model.fit(X, y)

print(f'first in-sample predictions: {iowa_model.predict(X.head())}')
print(f'actual target values: {y.head().tolist()}')
print(type(y.head()))

# now lets do that in the correct way : split the data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

# specify the model
iowa_model = DecisionTreeRegressor(random_state = 1)
# fit it
iowa_model.fit(train_X, train_y)

# predict
val_predictions = iowa_model.predict(val_X)

print(f'first in-sample predictions: {val_predictions[:10]}')
print(f'actual target values: {val_y[:10].tolist()}')

# let us calc the mae
val_mae = mean_absolute_error(val_y, val_predictions)
print(val_mae)

# make the function to estimate various tree sizes 
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

# write a loop to find the bes tree size
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

for max_leaf_nodes in candidate_max_leaf_nodes:
    mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print(f'for the max leaf nodes of {max_leaf_nodes} the mae equals: {mae}')

# fit model using the best tree size for the whole sample size
final_model = DecisionTreeRegressor(max_leaf_nodes = 100, random_state = 0)
final_model.fit(X, y)

# write the fuction that checks the mean absolute error on the tree
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# compare the sizes of the different leaf nodes 
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500, 1000]
for max_leaf_nodes in candidate_max_leaf_nodes:
    spam = []
    mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print(f'for the max leaf nodes of {max_leaf_nodes} the mean absolute error equals {mae}') 

# now use all the data you have with the best leaf nodes size 
final_model = DecisionTreeRegressor(max_leaf_nodes = 100, random_state=0)
final_model.fit(X, y)


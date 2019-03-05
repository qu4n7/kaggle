import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor 

melbourne_file_path = 'melb_data.csv'
melbourne_data = pd.read_csv('melb_data.csv')
print(melbourne_data.shape)
melbourne_data.columns

melbourne_data = melbourne_data.dropna(axis=0)
print(melbourne_data.shape)

y = melbourne_data.Price
print(y.shape)
print(y[:5])

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = melbourne_data[melbourne_features]
X.shape

X.describe()

X.head()

melbourne_model = DecisionTreeRegressor(random_state = 1) # specify a number for random_state to ensure same results each run
melbourne_model.fit(X, y)

print(f'predictions for the following 5 houses: {X.head()}')
print(f'the predicitons are {melbourne_model.predict(X.head())}')

print(melbourne_model.predict(X.head()))
print(y[:5])

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)

# now let's do that with the data split:

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# define model
melbourne_model = DecisionTreeRegressor()
# fit it
melbourne_model.fit(train_X, train_y)

# get predicted prices
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

# the next step is to avoid overfitiing / underfitting
# for the decision tree we might limit the number of the tree's leaf nodes:

# make the function

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return (mae)

# compare mae with different values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print(f'max leaf nodes: {max_leaf_nodes}, mean absolute error: {my_mae}')

# it is time for the random forest

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))


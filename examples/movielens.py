from tensorfm.sklearn import FactorizationMachineRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
import numpy as np

# Read in data
def load_data(filename, path="/Users/gmodena/Downloads/ml-100k/"):
    data = []
    y = []
    users=set()
    items=set()
    with open(path+filename) as f:
        for line in f:
            (user,movieid,rating,ts)=line.split('\t')
            data.append({ "user_id": str(user), "movie_id": str(movieid)})
            y.append(float(rating))
            users.add(user)
            items.add(movieid)

    return (data, np.array(y), users, items)

train_data, y_train, train_users, train_items = load_data("ua.base")
test_data, y_test, test_users, test_items = load_data("ua.test")
v = DictVectorizer()

X_train = v.fit_transform(train_data)
X_test = v.transform(test_data)

y_train.shape += (1,)

fm = FactorizationMachineRegressor(max_iter=10, n_factors=20, eta=0.01, C=10000, random_state=12345, penalty='l1')
fm.fit(X_train.todense(), y_train)
y_pred = fm.predict(X_test.todense())
print("MSE test ", mean_squared_error(y_test, y_pred))
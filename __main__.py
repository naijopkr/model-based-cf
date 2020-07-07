import numpy as np
import pandas as pd

column_names = ['user_id', 'item_id', 'rating', 'timestamp']

df = pd.read_csv('data/u.data', sep='\t', names=column_names)
df.head()

movie_titles = pd.read_csv('data/movie_id_titles.csv')
movie_titles.head()

df = pd.merge(df, movie_titles, on='item_id')
df.head()

n_users = df['user_id'].nunique()
n_items = df['item_id'].nunique()


# Memory-Based Collaborative Filtering

# Train test split
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(df, test_size=0.25)

# Create two user-item matrices, one for training and another
# for testing
def generate_matrix(data: pd.DataFrame):
    matrix = np.zeros((n_users, n_items))
    for row in data.itertuples():
        """
        Row example:

        Pandas(Index=43275, user_id=494, item_id=286, rating=4,
        timestamp=879540508, title='English Patient, The (1996)')

            row[1]-1 == 494 - 1 == 493 >> user_id - 1 for index
            row[2]-1 == 286 - 1 == 285 >> item_id - 1 for column
            row[3] == 4 >> rating for value
        """
        index = row[1] - 1
        column = row[2] - 1
        rating = row[3]
        matrix[index, column] = rating

    return matrix

train_data_matrix = generate_matrix(train_data)
test_data_matrix = generate_matrix(test_data)

# Sparsity for MovieLens100k is 93.7%
sparsity = round(1.0 - len(df)/float(n_users*n_items), 3)

import scipy.sparse as sp
from scipy.sparse.linalg import svds
from math import sqrt
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    prediction = y_pred[y_true.nonzero()].flatten()
    ground_truth = y_true[y_true.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


# Get SVD components from train matrix. Choose k.
u, s, vt = svds(train_data_matrix, k = 20)
s_diag_matrix = np.diag(s)
X_pred = np.dot(np.dot(u, s_diag_matrix), vt)

pred_rmse = rmse(test_data_matrix, X_pred) # 2.719

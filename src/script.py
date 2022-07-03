# %% [markdown]
# # SYMA

# %%
import pandas as pd
import os

# %% [markdown]
# ## Loading Data

# %%
PATH_DATA = os.path.join("..", "data")

train_session_df = pd.read_csv(os.path.join(PATH_DATA, "train_sessions.csv"))
train_purchase_df = pd.read_csv(os.path.join(PATH_DATA, "train_purchases.csv"))

candidate_items_df = pd.read_csv(os.path.join(PATH_DATA, "candidate_items.csv"))
item_features_df = pd.read_csv(os.path.join(PATH_DATA, "item_features.csv"))

# %%
train_session_df.describe()

# %%
train_purchase_df.describe()

# %%
candidate_items_df.describe()

# %%
item_features_df.describe()

# %% [markdown]
# ## Data exploration

# %% [markdown]
# *How many different items does exist?*

# %%
distinct_item_number = len(item_features_df.item_id.unique())
print("Unique item number :", distinct_item_number)
print(
    "Item id are unique : ",
    item_features_df.item_id.nunique() == len(item_features_df.item_id.unique()),
)

# %% [markdown]
# *How many different sessions does exist?*

# %%
distinct_session_number = len(
    pd.concat([train_session_df.session_id, train_purchase_df.session_id]).unique()
)
print("Unique user number :", distinct_session_number)

# %% [markdown]
# *Does session always look an item before buying it?*

# %%
import numpy as np

print("A user never look at one item before buying it.")
pd.merge(
    train_purchase_df,
    train_session_df,
    on=["session_id", "item_id"],
    how="left",
    indicator="Exist",
)["Exist"].value_counts()

# %% [markdown]
# *Can a session look at items without buying any?*

# %%
print("Every session bought exactly one item.")

pd.merge(
    train_purchase_df,
    train_session_df,
    on=["session_id"],
    how="left",
    indicator="Exist",
)["Exist"].value_counts()

# %% [markdown]
# *What is the average number of different items every user usually look?*

# %%
print(
    "Average number of items seen by user :",
    train_session_df.groupby("session_id").count()["item_id"].mean(),
)

# %% [markdown]
# *What will be the size of our rating matrix?*

# %%
print(
    "Size of the maximum full rating matrix : ",
    (distinct_session_number * distinct_item_number, 3),
)

# %% [markdown]
# ## SVD++

# %% [markdown]
# We want to create ratings given by every session for every item. We will first choose the following rating system:
# - If the user has seen the item, we will give it a rating of 1.
# - If the user purchased the item, we will give it a rating of 2.

# %%
# ----------------------------- WE CREATE RATINGS ---------------------------- #
train_rating_df = pd.concat(
    [train_session_df.assign(rating=1), train_purchase_df.assign(rating=2)]
)
train_rating_df.describe()

# %%
# ---------------------------- SHUFFLE AND RENAME ---------------------------- #
train_ratings_df_shuffled = train_rating_df.sample(len(train_rating_df))
train_ratings_df_shuffled.rename(
    columns={"session_id": "user_id", "rating": "raw_ratings"}, inplace=True
)

# %%
# -------------- WE REDUCE THE SIZE OF OUR DATASET FOR RESEARCH -------------- #

train_set_df_reduced = train_ratings_df_shuffled[:10000]

# %%
# ----------------------------- WE CREATE OUR SET ---------------------------- #

import surprise

rating_reader = surprise.Reader(rating_scale=(1, 2))
dataset = surprise.dataset.Dataset.load_from_df(
    df=train_set_df_reduced[["user_id", "item_id", "raw_ratings"]], reader=rating_reader
)

# %%
import surprise
import sklearn.model_selection

train_set_df, test_set_df = sklearn.model_selection.train_test_split(
    train_set_df_reduced
)

rating_reader = surprise.Reader(rating_scale=(1, 2))
train_set = surprise.dataset.Dataset.load_from_df(
    df=train_set_df[["user_id", "item_id", "raw_ratings"]], reader=rating_reader
)
test_set = surprise.dataset.Dataset.load_from_df(
    df=test_set_df[["user_id", "item_id", "raw_ratings"]], reader=rating_reader
)

# %%
# ------------------------- WE TRAIN OUR FIRST MODEL ------------------------- #

model = surprise.SVD()

surprise.model_selection.cross_validate(
    model, train_set, measures=["RMSE", "MAE"], cv=5, verbose=True, n_jobs=-1
)

# %% [markdown]
# # Let's compare our models

# %%
model_list = [
    surprise.NormalPredictor(),
    surprise.BaselineOnly(),
    surprise.KNNBaseline(),
    surprise.KNNBasic(),
    surprise.KNNWithMeans(),
    surprise.KNNWithZScore(),
    surprise.SlopeOne(),
    surprise.SVD(),
    surprise.SVDpp(),
    surprise.NMF(),
    surprise.CoClustering(),
    surprise.SlopeOne(),
]

result = {}
for model in model_list:
    scores = surprise.model_selection.cross_validate(
        model, train_set, measures=["RMSE", "MAE"], cv=5, verbose=False
    )
    result[model.__class__.__name__] = (
        scores["test_rmse"].mean(),
        scores["test_mae"].mean(),
    )

# %%
# ------------------------- BEST ALGORITHMS WITH RMSE ------------------------ #
sorted(result.items(), key=lambda x: x[1][0])

# %%
# ------------------------- BEST ALGORITHMS WITH MAE ------------------------- #
sorted(result.items(), key=lambda x: x[1][1])

# %%
# ------------------------------ BEST ALGORITHMS ----------------------------- #
import numpy as np

sorted(result.items(), key=lambda x: np.mean(x[1]))

# %% [markdown]
# ## Now perform some Grid Search

# %%
# ------------------------- CREATING CROSS VALIDATION ------------------------ #

import sklearn.model_selection


class MyCrossValidation:
    def __init__(self, params):
        self.SVD_list = [
            (surprise.SVD(**args, verbose=False), args)
            for args in list(sklearn.model_selection.ParameterGrid(params))
        ]
        self.full_train_set = train_set.build_full_trainset()

    def __train_test_model(self, svd_model, params, verbose=1):
        svd_model.fit(
            self.full_train_set,
        )
        predictions = svd_model.test(self.full_train_set.build_testset())
        score = surprise.accuracy.rmse(
            predictions, verbose=True if verbose == 2 else False
        )
        if verbose == 1:
            print("Params {} :".format(str(params)), score)
        return (params, score)

    def __call__(self, verbose=1):
        res = []
        i = 0
        while len(self.SVD_list):
            svd_model, params = self.SVD_list.pop()
            i += 1
            if verbose == 1:
                print("{}/{}".format(i, len(self.SVD_list)), end=" --- ")
            res.append(self.__train_test_model(svd_model, params, verbose))
            del svd_model
        return sorted(res, key=lambda x: x[2])


# %%
# ----------------------------- FIND BEST PARAMS ----------------------------- #
params = {
    "biased": [True, False],
    "init_std_dev": [0.01, 0.1, 1],
    "lr_all": [0.001],
    "reg_bu": [0.001, 0.01, 0.1],
    "reg_bi": [0.001, 0.01, 0.1],
    "reg_qi": [0.001, 0.01, 0.1],
    "reg_pu": [0.001, 0.01, 0.1],
}

best_model = MyCrossValidation(params)()


print(best_model[0][1], best_model[0][2])

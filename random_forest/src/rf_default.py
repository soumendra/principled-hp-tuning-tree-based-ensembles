import time

import pandas as pd  # type:ignore
from dataset import Dataset  # type:ignore
from sklearn.compose import ColumnTransformer  # type:ignore
from sklearn.ensemble import RandomForestClassifier  # type:ignore
from sklearn.model_selection import RandomizedSearchCV  # type:ignore
from sklearn.pipeline import Pipeline  # type:ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type:ignore

best_models = []
all_models = pd.DataFrame()

d = Dataset()

for name in d.openml.name.values:
    start = time.time()
    print(f"Starting to train dataset: {d.name}")
    try:
        d = d.fetch_dataset(name)
    except:
        print(f"Dataset download failed for {d.name}")
        continue
    print(f"Downloaded dataset has {d.x.shape[0]} rows and {d.x.shape[1]} columns")
    print(f"Length of the target variable: {len(d.y)}")
    print(f"Column types: {d.x.dtypes}")

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    numeric_transformer = StandardScaler()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, d.numerical_ix),
            ("cat", categorical_transformer, d.categorical_ix),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier()),
        ]
    )

    random_searcher = RandomizedSearchCV(
        pipeline,
        n_iter=1,
        param_distributions={
            "model__n_jobs": [-1],
            "model__random_state": [2020],
        },
        n_jobs=-1,
        cv=7,
        random_state=2020,
        scoring="accuracy",
    )
    try:
        random_searcher.fit(d.X_train, d.y_train)
    except:
        print(f"Model training failed for {d.name}")
        continue
    end = time.time()

    best_models += [
        {
            "name": d.name,
            "time_elapsed": end - start,
            "best_score": random_searcher.best_score_,
            "test_score": random_searcher.score(d.X_test, d.y_test),
            **random_searcher.best_params_,
        }
    ]
    best_models_pd = pd.DataFrame.from_records(best_models).drop(
        ["model__random_state", "model__n_jobs"], axis=1
    )
    best_models_pd.to_csv("best_models.csv")

    new_model = pd.DataFrame(random_searcher.cv_results_)
    new_model["name"] = d.name
    all_models = all_models.append(new_model)
    all_models.drop(
        ["params", "param_model__random_state", "param_model__n_jobs"], axis=1
    ).to_csv("all_models.csv")

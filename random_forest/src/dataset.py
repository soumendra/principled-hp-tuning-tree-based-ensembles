from typing import Tuple

import attr
import numpy as np  # type:ignore
import pandas as pd  # type:ignore
from datalists import openml_df
from sklearn.datasets import fetch_openml  # type:ignore
from sklearn.model_selection import train_test_split  # type:ignore


@attr.s
class Dataset:
    name: str = attr.ib(default="")
    x: pd.DataFrame = attr.ib(default=None)
    y: np.ndarray = attr.ib(default=None)
    openml: pd.DataFrame = attr.ib(default=openml_df)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> Tuple:
        return (self.x[i], self.y[i])

    def describe(self):
        return self.x.describe()

    def fetch_dataset(self, name: str = ""):
        # There is also this (cache with openml): https://pplonski.github.io/python-and-pandas-reading-data-from-openml/
        openml_row = self.openml[self.openml.name == name]
        if openml_row.shape[0] == 0:
            print(f"[ALERT] No matching record for the provided name ({name}) found.")
        else:
            print(f"[ALERT] matching record for the provided name ({name}) found. Downloading data...")
            self.name = name
            # This needs some error handling
            self.x, self.y = fetch_openml(
                data_id=openml_row.openml_dataset_id.values[0],
                data_home="~/data-zoo/sklearn",
                target_column=openml_row.target.values[0],
                cache=True,
                return_X_y=True,
                as_frame=True,
            )
            print("Download complete.")
            id_cols = openml_row.row_id_col.values[0]
            if id_cols is not None:
                for id_col in id_cols:
                    if id_col in self.x.columns:
                        self.x = self.x.drop(id_col, axis=1)
                        print(f"Dropped column: {id_col}")

            ignore_cols = openml_row.ignore_cols.values[0]
            if ignore_cols is not None:
                for ignore_col in ignore_cols:
                    if ignore_col in self.x.columns:
                        self.x = self.x.drop(ignore_col, axis=1)
                        print(f"[ALERT] Dropped column: {ignore_col} from dataset: {openml_row.name.values[0]}")

            self.numerical_ix = self.x.select_dtypes(include=[np.number, "int64", "float64"]).columns
            self.categorical_ix = self.x.select_dtypes(include=["object", "bool", "category"]).columns
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.x, self.y, test_size=0.2, random_state=2020, stratify=self.y
            )
        return self

import pandas as pd
import numpy as np

def load_and_clean_data(filepath):
    data = pd.read_csv(filepath)

    # Sort by week
    data = data.sort_values("week")

    # Create synthetic target variable
    data["estimated_demand"] = (
        (1000 / data["checkout_price"]) +
        (data["emailer_for_promotion"] * 200) +
        (data["homepage_featured"] * 300)
    )

    # Feature selection
    features = [
        "week",
        "checkout_price",
        "base_price",
        "emailer_for_promotion",
        "homepage_featured"
    ]

    X = data[features]
    y = data["estimated_demand"]

    return data, X, y


if __name__ == "__main__":
    df, X, y = load_and_clean_data("../dataset/food_Demand_test.csv")
    print(df.head())

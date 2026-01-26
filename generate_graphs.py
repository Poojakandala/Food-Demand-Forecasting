import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing.data_cleaning import load_and_clean_data
from models.regression_models import train_models
from sklearn.model_selection import train_test_split

# Load data
data, X, y = load_and_clean_data("../dataset/food_Demand_test.csv")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Train model
models = train_models(X_train, y_train)
model = models["Linear Regression"]

# Predictions
data["prediction"] = model.predict(X)
data["residuals"] = data["estimated_demand"] - data["prediction"]

fig, axes = plt.subplots(3, 2, figsize=(16, 12))

# 1. Actual vs Predicted
axes[0, 0].plot(data["week"], data["estimated_demand"], label="Actual")
axes[0, 0].plot(data["week"], data["prediction"], label="Predicted")
axes[0, 0].set_title("Actual vs Predicted Demand")
axes[0, 0].legend()

# 2. Scatter plot
axes[0, 1].scatter(data["estimated_demand"], data["prediction"])
axes[0, 1].set_title("Actual vs Predicted Scatter")
axes[0, 1].set_xlabel("Actual Demand")
axes[0, 1].set_ylabel("Predicted Demand")

# 3. Residuals
axes[1, 0].plot(data["week"], data["residuals"])
axes[1, 0].axhline(0, color="red", linestyle="--")
axes[1, 0].set_title("Residuals Over Time")

# 4. Box plot
sns.boxplot(y=data["estimated_demand"], ax=axes[1, 1])
axes[1, 1].set_title("Demand Distribution")

# 5. Heatmap
sns.heatmap(
    data[
        ["checkout_price", "base_price",
         "emailer_for_promotion", "homepage_featured",
         "estimated_demand"]
    ].corr(),
    annot=True,
    cmap="coolwarm",
    ax=axes[2, 0]
)
axes[2, 0].set_title("Correlation Heatmap")

axes[2, 1].axis("off")

plt.tight_layout()
plt.savefig("graphs.png")
plt.show()

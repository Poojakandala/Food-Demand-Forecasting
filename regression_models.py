from sklearn.linear_model import LinearRegression, Ridge, Lasso

def train_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.01)
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models
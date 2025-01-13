
from mlflow_train import test_train_split, load_data, create_model


data = load_data()

def test_train_test_split():
    X_train, X_test, y_train, y_test = test_train_split(data)
    assert X_train.shape[1] == X_test.shape[1], "Train and test feature dimensions mismatch"

def test_model_accuracy():
    X_train, X_test, y_train, y_test = test_train_split(data)
    model = create_model(n_estimators,random_state)
    model.fit(X_train, y_train.values.ravel())
    accuracy = model.score(X_test, y_test)
    assert accuracy > 0.8, "Model accuracy is below acceptable threshold"

n_estimators  = 20
random_state = 42
test_train_test_split()
test_model_accuracy()



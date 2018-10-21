import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def main():
    dataset = load_diabetes().data

    X = dataset[:,0:8]
    Y = dataset[:,8]

    seed = 7
    test_size = 0.33
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    model = XGBClassifier()
    model.fit(x_train, y_train)

    mlflow.set_experiment("default", artifact_relative=True)
    mlflow.start_run(artifact_relative=True)
    mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    main()

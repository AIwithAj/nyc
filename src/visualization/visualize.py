import pathlib
import joblib
import sys
import numpy as np
import yaml
import mlflow
import pandas as pd
from sklearn import metrics
from sklearn import tree
from dvclive import Live
from matplotlib import pyplot as plt


def evaluate(model, X, y, split, live, save_path):
    """
    Dump all evaluation metrics and plots for given datasets.

    Args:
        model (sklearn.ensemble.RandomForestClassifier): Trained classifier.
        X (pandas.DataFrame): Input DF.
        y (pamdas.Series): Target column.
        split (str): Dataset name.
        live (dvclive.Live): Dvclive instance.
        save_path (str): Path to save the metrics.
    """

    
    predictions = model.predict(X)

    # Use dvclive to log a few simple metrics...
    # avg_prec = metrics.average_precision_score(y, predictions)
    r2score=metrics.r2_score(y,predictions)
    # roc_auc = metrics.roc_auc_score(y, predictions)
    rmse=np.sqrt(metrics.mean_squared_error(y,predictions))
    if not live.summary:
        live.summary = {"r2": {}, "rmse": {}}
    live.summary["r2"][split] = r2score
    live.summary["rmse"][split] = rmse
    mlflow.log_metric(key="r2", value=r2score)
    mlflow.log_metric(key="rmse", value=rmse)
    live.log_metric(f"{split}/r2", r2score)
    live.log_metric(f"{split}/rmse",rmse)



    # ... and plots...
    # ... like an roc plot...
    # live.log_sklearn_plot("r2_score", y, predictions, name=f"r2/{split}")
    # ... and precision recall plot...
    # ... which passes `drop_intermediate=True` to the sklearn method...
    # live.log_sklearn_plot(
    #     "precision_recall",
    #     y,
    #     predictions,
    #     name=f"prc/{split}",
    #     drop_intermediate=True,
    # )
    # ... and confusion matrix plot
    # live.log_sklearn_plot(
    #     "confusion_matrix",
    #     y,
    #     predictions_by_class.argmax(-1),
    #     name=f"cm/{split}",
    # )


def save_importance_plot(live, model, feature_names):
    """
    Save feature importance plot.

    Args:
        live (dvclive.Live): DVCLive instance.
        model (sklearn.ensemble.RandomForestClassifier): Trained classifier.
        feature_names (list): List of feature names.
    """
    fig, axes = plt.subplots(dpi=100)
    fig.subplots_adjust(bottom=0.2, top=0.95)
    axes.set_ylabel("Mean decrease in impurity")

    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names).nlargest(n=10)
    forest_importances.plot.bar(ax=axes)

    live.log_image("importance.png", fig)


def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    # TODO - Optionally add visualization params as well
    # params_file = home_dir.as_posix() + '/params.yaml'
    # params = yaml.safe_load(open(params_file))["train_model"]

    model_file = home_dir.as_posix() + sys.argv[1]
    # Load the model.
    model = joblib.load(model_file)
    
    # Load the data.
    input_file = sys.argv[2]
    data_path = home_dir.as_posix() + input_file
    output_path = home_dir.as_posix() + '/dvclive'
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    TARGET = 'trip_duration'
    train_features = pd.read_csv(data_path + '/train.csv')
    X_train = train_features.drop([TARGET,'id'], axis=1)
    y_train = train_features[TARGET]
    feature_names = X_train.columns.to_list()

    test_features = pd.read_csv(data_path + '/test.csv')
    X_test = test_features.drop([TARGET,'id'], axis=1)

    y_test = test_features[TARGET]

    # Evaluate train and test datasets.
    with Live(output_path,save_dvc_exp=True,dvcyaml=True) as live:
        evaluate(model, X_train, y_train, "train", live, output_path)
        evaluate(model, X_test, y_test, "test", live, output_path)

        # Dump feature importance plot.
        save_importance_plot(live, model, feature_names)

if __name__ == "__main__":
    main()

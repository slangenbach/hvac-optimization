from pathlib import Path
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import f1_score, classification_report, plot_confusion_matrix, plot_precision_recall_curve


def save_fig(plot, figure_path: Path, figure_name: str) -> None:
    """
    Save matplotlib plot to PNG file on local disk.
    
    """
    # get subplot object if it exists
    if type(plot) == np.ndarray:
        plot = plot[0]
        
    plot.get_figure().savefig(figure_path.joinpath(f"{figure_name}.png"))


def run_pipeline(pipe, X_train, y_train, X_test, y_test, features, model_path, model_name, save_model=False) -> dict:
    """
    Run MLflow pipeline: Train model, predict on train/test data, calculate cross-validated F1 scores, plot results
    and log params, metrics, artifacts and model.
    
    """
    with mlflow.start_run():
        
        # define pipeline, fit model and predict
        print(f"Running pipeline for '{model_name}' model")
        model = pipe.fit(X_train[features], y_train)
        y_pred_train = model.predict(X_train[features])
        y_pred_test = model.predict(X_test[features])
    
        # calculate metrics
        f1_train = f1_score(y_train, y_pred_train)
        f1_test = f1_score(y_test, y_pred_test)
        cv_score_train = np.average(cross_val_score(model, X_train[features], y_pred_train,
                                             scoring="f1", cv=5, n_jobs=-1))
        cv_score_test = np.average(cross_val_score(model, X_test[features], y_pred_test, 
                                              scoring="f1", cv=5, n_jobs=-1))
        
        # print results
        print(f"CV F1-score train: {np.round(cv_score_train, 4)}", 
              f"CV F1-score test: {np.round(cv_score_test, 4)}", sep="\n")
    
        # visualize model performance
        print("Classification report")
        print(classification_report(y_test, y_pred_test))
        
        if hasattr(model.named_steps["model"], "feature_importances_"):
            feat_importances = get_feature_importances(features, model.named_steps["model"])
            feat_imp_plot = plot_feature_importances(feat_importances)
            save_fig(feat_imp_plot, model_path, f"{model_name}_feat_importances_plot")  # TODO: Change path
            #mlflow.log_artifact(model_path.joinpath(f"{model_name}_feat_importances_plot.png"))
          
        # log params, metrics, artifacts and model
        mlflow.log_params(model.named_steps["model"].get_params())
        mlflow.log_metrics({"f1_score_train": f1_train, "cross_val_score_train": cv_score_train, 
                            "f1_score_test": f1_test, "cross_val_score_test": cv_score_test})
        mlflow.sklearn.log_model(model, model_name)
        
        if save_model:
            mlflow.sklearn.save_model(model, model_path.joinpath(model_name))
        
        result = {"model": model, "y_pred": y_pred_test}
        return result

def get_feature_importances(features: list, model) -> list:
    """
    Return feature importances as sorted list of lists -> [["feature_1", 0.42], ["feature_2", 0.1337], ...]
    
    """
    return sorted([e for e in zip(features, model.feature_importances_)], key=lambda x: x[1], reverse=True)


def plot_feature_importances(feature_importances: list):
    """
    Plot feature importances as horizontal barchart
    
    """
    df = pd.DataFrame.from_records(feature_importances, columns=["feature", "importance"])
    plot = df.plot(kind="barh", x="feature", y="importance", figsize=(8.3, 5.8), title="Feature importances")
    
    return plot
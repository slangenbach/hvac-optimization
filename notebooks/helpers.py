from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib as mpl
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, plot_confusion_matrix, accuracy_score, f1_score


def save_fig(plot, figure_path: Path, figure_name: str, file_format: str = "svg", dpi: int = 150) -> None:
    """
    Save matplotlib plot to file on local disk.
    
    """
    # get subplot object if it exists
    if type(plot) == np.ndarray:

        # get nested subplot, e.g. if layout contains two columns
        if type(plot[0]) == np.ndarray:
            plot = plot[0][0]
        else:
            plot = plot[0]

    plot.get_figure().savefig(figure_path.joinpath(f"{figure_name}.{file_format}"), format=file_format, dpi=dpi)


class MLPipeline:
    """
    Wrapper class for running MLflow boosted sklearn pipelines.
    """

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
                 model_path: Path, figure_path: Path, mlflow_path: str, mlflow_experiment: str):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_path = model_path
        self.figure_path = figure_path
        self.mlflow_path = mlflow_path
        self.mlflow_experiment = mlflow_experiment

        # config Mlflow
        mlflow.set_tracking_uri(self.mlflow_path)
        mlflow.set_experiment(self.mlflow_experiment)

        # use seaborn styling for plots
        mpl.style.use("seaborn-colorblind")
        mpl.rcParams.update({
            # set figure size to rotated A5 paper size for all plots
            "figure.figsize": (5.8, 4.1),
            "figure.titlesize": "large",
            "axes.labelsize": "medium",
            "axes.titlesize": "x-large",
            "xtick.labelsize": "small",
            "ytick.labelsize": "small",
            "axes.grid": False})

    def run_pipeline(self, pipe, features: list, model_name: str, cv: int = 5, save_model: bool = False) -> dict:
        """
        Run MLflow pipeline: Train model, predict on train/test data, calculate cross-validated F1 scores, plot results
        and log params, metrics, artifacts and model.

        :param pipe: sklearn Pipeline
        :param features: List of features used for model
        :param model_name: Name of model running through pipeline
        :param cv: Cross-validation parameter used in sklearn cross_val_score method
        :param save_model: Boolean indicating whether to save model to disk
        :returns: Dict containing sklearn model object, predictions on self.X_train and self.X_test, and missclassifications
        """
        with mlflow.start_run():

            # define pipeline, fit model and predict
            print(f"Running pipeline for '{model_name}' model")
            model = pipe.fit(self.X_train[features], self.y_train)
            y_pred_train = model.predict(self.X_train[features])
            y_pred_test = model.predict(self.X_test[features])

            # get missclassifications
            misses_train = self.X_train[features][self.y_train != y_pred_train].index
            misses_test = self.X_test[features][self.y_test != y_pred_test].index

            # calculate (cross-validated) metrics
            acc_train = accuracy_score(self.y_train, y_pred_train)
            acc_test = accuracy_score(self.y_test, y_pred_test)
            f1_train = f1_score(self.y_train, y_pred_train)
            f1_test = f1_score(self.y_test, y_pred_test)

            cv_acc_train = np.average(cross_val_score(model, self.X_train[features], y_pred_train,
                                                      scoring="accuracy", cv=cv, n_jobs=-1))
            cv_acc_test = np.average(cross_val_score(model, self.X_test[features], y_pred_test,
                                                     scoring="accuracy", cv=cv, n_jobs=-1))
            cv_f1_train = np.average(cross_val_score(model, self.X_train[features], y_pred_train,
                                                     scoring="f1", cv=cv, n_jobs=-1))
            cv_f1_test = np.average(cross_val_score(model, self.X_test[features], y_pred_test,
                                                    scoring="f1", cv=cv, n_jobs=-1))

            # print results
            print(
                f"Accuracy train: {np.round(acc_train, 4)}",
                f"Accuracy test: {np.round(acc_test, 4)}",
                f"F1-score train: {np.round(f1_train, 4)}",
                f"F1-score test: {np.round(f1_test, 4)}",
                f"CV accuracy train: {np.round(cv_acc_train, 4)}",
                f"CV accuracy test: {np.round(cv_acc_test, 4)}",
                f"CV F1-score train: {np.round(cv_f1_train, 4)}",
                f"CV F1-score test: {np.round(cv_f1_test, 4)}",
                sep="\n")

            print("Classification report")
            print(classification_report(self.y_test, y_pred_test))

            # print confusion matrix
            if not "encode" in model.named_steps.keys():
                plot_confusion_matrix(model.named_steps["model"], self.X_test[features], self.y_test, cmap="Blues",
                                      normalize=None, values_format="d")

            # print feature importances
            if hasattr(model.named_steps["model"], "feature_importances_"):
                feat_importances = self._get_feature_importances(features, model.named_steps["model"])
                feat_imp_plot = self._plot_feature_importances(feat_importances)
                save_fig(feat_imp_plot, self.figure_path, f"{model_name}_feat_importances_plot")
                mlflow.log_artifact(self.figure_path.joinpath(f"{model_name}_feat_importances_plot.svg"))

            # log params, metrics, artifacts and model
            mlflow.log_params(model.named_steps["model"].get_params())
            mlflow.log_metrics({
                "cv_accuracy_train": cv_acc_train,
                "cv_accuracy_test": cv_acc_train,
                "cv_f1_score_train": cv_f1_train,
                "cv_f1_score_test": cv_f1_train})
            mlflow.sklearn.log_model(model, model_name)

            if save_model:
                mlflow.sklearn.save_model(model, self.model_path.joinpath(model_name))

            result = {"model": model, "y_pred_test": y_pred_test, "y_pred_train": y_pred_train,
                      "misses_train": misses_train, "misses_test": misses_test}

            return result

    @staticmethod
    def _get_feature_importances(features: list, model) -> list:
        """
        Return feature importances as sorted list of lists -> [["feature_1", 0.42], ["feature_2", 0.1337], ...]
        
        :param features: List of feature names
        :param model: sklearn model
        :returns: Sorted list of feature names and corresponding importances
        """
        return sorted([e for e in zip(features, model.feature_importances_)], key=lambda x: x[1], reverse=True)

    @staticmethod
    def _plot_feature_importances(feature_importances: list):
        """
        Plot feature importances as horizontal barchart
        
        :param feature_importances: List of feature importances produced by certain sklearn models
        :returns: Barplot of feature importances
        """
        df = pd.DataFrame.from_records(feature_importances, columns=["feature", "importance"])
        plot = df.plot(kind="barh", x="feature", y="importance", legend=False, grid=True, title="Feature importances")
        plot.set_xlabel("importance")

        return plot

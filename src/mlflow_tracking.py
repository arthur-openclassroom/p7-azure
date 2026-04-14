"""Fonction standardisée de tracking MLFlow pour tous les modèles du projet."""

import time
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)


def log_training_run(
    run_name: str,
    y_true,
    y_pred,
    y_proba,
    params: dict,
    fit_time: float,
    predict_time: float,
    model=None,
    model_type: str = "sklearn",
    history=None,
    register_name: str | None = None,
):
    """
    Logue de manière standardisée une expérience dans MLFlow.

    Parameters
    ----------
    run_name : str
        Nom du run MLFlow.
    y_true : array-like
        Labels réels.
    y_pred : array-like
        Prédictions binaires (0/1).
    y_proba : array-like
        Probabilités de la classe positive.
    params : dict
        Hyperparamètres à loguer.
    fit_time : float
        Temps d'entraînement en secondes.
    predict_time : float
        Temps de prédiction en secondes.
    model : object, optional
        Modèle entraîné à loguer dans MLFlow.
    model_type : str
        "sklearn", "tensorflow" ou "transformers".
    history : dict, optional
        Historique Keras (history.history) pour les courbes d'apprentissage.
    register_name : str, optional
        Nom sous lequel enregistrer le modèle dans le Model Registry.

    Returns
    -------
    run_id : str
        ID du run MLFlow créé.
    metrics : dict
        Dictionnaire des métriques calculées.
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "fit_time_seconds": fit_time,
            "predict_time_seconds": predict_time,
        }
        mlflow.log_metrics(metrics)

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        ax_roc.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.4f}")
        ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax_roc.set_xlabel("Taux de faux positifs")
        ax_roc.set_ylabel("Taux de vrais positifs")
        ax_roc.set_title(f"Courbe ROC — {run_name}")
        ax_roc.legend()
        ax_roc.grid(True, alpha=0.3)
        plt.tight_layout()
        mlflow.log_figure(fig_roc, "roc_curve.png")
        plt.close(fig_roc)

        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred,
            display_labels=["Négatif", "Positif"],
            ax=ax_cm,
        )
        ax_cm.set_title(f"Matrice de confusion — {run_name}")
        plt.tight_layout()
        mlflow.log_figure(fig_cm, "confusion_matrix.png")
        plt.close(fig_cm)

        if history is not None:
            fig_hist, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].plot(history["accuracy"], label="Train")
            axes[0].plot(history["val_accuracy"], label="Validation")
            axes[0].set_title("Accuracy par époque")
            axes[0].set_xlabel("Époque")
            axes[0].set_ylabel("Accuracy")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[1].plot(history["loss"], label="Train")
            axes[1].plot(history["val_loss"], label="Validation")
            axes[1].set_title("Loss par époque")
            axes[1].set_xlabel("Époque")
            axes[1].set_ylabel("Loss")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            plt.suptitle(f"Courbes d'apprentissage — {run_name}", fontsize=13)
            plt.tight_layout()
            mlflow.log_figure(fig_hist, "history_plot.png")
            plt.close(fig_hist)

        if model is not None:
            if model_type == "sklearn":
                mlflow.sklearn.log_model(model, "model")
            elif model_type == "tensorflow":
                mlflow.tensorflow.log_model(model, "model")

            if register_name:
                run_id = mlflow.active_run().info.run_id
                mlflow.register_model(f"runs:/{run_id}/model", register_name)

        run_id = mlflow.active_run().info.run_id

    return run_id, metrics

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def plot_roc_and_prc_curve(metrics, title, level):
    """
    Plots the ROC (Receiver Operating Characteristic) and PRC (Precision-Recall Curve) for a binary classifier, baseline curves are included for reference.
    Args:
        metrics (dict): A dictionary containing the following keys:
            - fpr_curve (array-like): False positive rates for the ROC curve.
            - tpr_curve (array-like): True positive rates for the ROC curve.
            - precision_curve (array-like): Precision values for the PRC curve.
            - recall_curve (array-like): Recall values for the PRC curve.
            - roc_auc_score (float): Area Under the Curve (AUC) score for the ROC curve.
            - prc_auc_score (float): Area Under the Curve (AUC) score for the PRC curve.
            - tp (int): Number of true positives.
            - fp (int): Number of false positives.
            - tn (int): Number of true negatives.
            - fn (int): Number of false negatives.
        title (str): Title for the plot, typically indicating the dataset or split (e.g., "Test", "Validation").
        level (str): Level of analysis or model (e.g., "Image", "Patient").
    """
    fpr_curve = metrics['fpr_curve']
    tpr_curve = metrics['tpr_curve']
    precision_curve = metrics['precision_curve']
    recall_curve = metrics['recall_curve']
    roc_auc_score = metrics['roc_auc_score']
    prc_auc_score = metrics['prc_auc_score']
    tp = metrics['tp']
    fp = metrics['fp']
    tn = metrics['tn']
    fn = metrics['fn']

    prc_base_value = (tp + fn) / (tp + fn + fp + tn) if (tp + fn + fp + tn) > 0 else 0.0

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fpr_curve, tpr_curve, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Baseline (AUC = 0.5)')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{level} Level - ROC Curve on the {title} Set")
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(recall_curve, precision_curve, color='darkgreen', lw=2, label=f'PRC curve (AUC = {prc_auc_score:.4f})')
    plt.plot([0, 1], [prc_base_value, prc_base_value], color='navy', lw=2, linestyle='--', label=f'Baseline (AUC = {prc_base_value:.4f})')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"{level} Level - PRC Curve on the {title} Set")
    plt.legend(loc="lower left")
    plt.grid(True)

    plt.show()

def plot_roc_and_prc_curves_together(all_metrics, title, level):
    """
    Plots ROC and PRC curves for multiple models on the same graph for comparison.
    Args:
        all_metrics (list of dict): A list where each element is a metrics dictionary for a model.
        title (str): Title for the plot, typically indicating the dataset or split (e.g., "Test", "Validation").
        level (str): Level of analysis or model (e.g., "Texture", "Feature").
    """
    plt.figure(figsize=(16, 6))

    # ROC Curve
    plt.subplot(1, 2, 1)
    for i, metrics in enumerate(all_metrics):
        fpr_curve = metrics['fpr_curve']
        tpr_curve = metrics['tpr_curve']
        roc_auc_score = metrics['roc_auc_score']
        plt.plot(fpr_curve, tpr_curve, lw=2, label=f'Model {i+1} ROC curve (AUC = {roc_auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Baseline (AUC = 0.5)')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{level} Level - ROC Curve on the {title} Set")
    plt.legend(loc="lower right")
    plt.grid(True)

    # PRC Curve
    plt.subplot(1, 2, 2)
    prc_base_value = 0.0
    for i, metrics in enumerate(all_metrics):
        precision_curve = metrics['precision_curve']
        recall_curve = metrics['recall_curve']
        prc_auc_score = metrics['prc_auc_score']
        tp = metrics['tp']
        fp = metrics['fp']
        tn = metrics['tn']
        fn = metrics['fn']
        if i == 0:
            prc_base_value = (tp + fn) / (tp + fn + fp + tn) if (tp + fn + fp + tn) > 0 else 0.0
        plt.plot(recall_curve, precision_curve, lw=2, label=f'Model {i+1} PRC curve (AUC = {prc_auc_score:.4f})')
    plt.plot([0, 1], [prc_base_value, prc_base_value], color='navy', lw=2, linestyle='--', label=f'Baseline (AUC = {prc_base_value:.4f})')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"{level} Level - PRC Curve on the {title} Set")
    plt.legend(loc="lower left")
    plt.grid(True)

    plt.show()
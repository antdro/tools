import seaborn as sb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def get_cm_in_proc(cm):
    tp, fn, fp, tn = cm.ravel()
    tp_proc = round(100 * tp / (tp + fn), 1)
    fn_proc = round(100 * fn / (tp + fn), 1)
    fp_proc = round(100 * fp / (fp + tn), 1)
    tn_proc = round(100 * tn / (fp + tn), 1)

    return tp_proc, fn_proc, fp_proc, tn_proc

def confusion_matrix_comparison_plot(actual, model_1, model_2):

    fig_title = 'Confusion Matrix for Prediction'
    model_1_title = 'Model 1'
    model_2_title = 'Model 2'
    positive_label = 'positive'
    negative_label = 'negative'
    x_label = 'Predicted'
    y_label = 'True'

    colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
    color_map = sb.xkcd_palette(colors)
    
    cm_model_1 = confusion_matrix(actual, model_1, labels = [1, 0])
    cm_model_2 = confusion_matrix(actual, model_2, labels = [1, 0])
    cm_model_1_proc = get_cm_in_proc(cm_model_1)
    cm_model_2_proc = get_cm_in_proc(cm_model_2)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sb.heatmap(np.reshape(cm_model_1_proc, (2,2)), annot=True, ax=axes[0], fmt='g', cmap=color_map)
    for pos, t in enumerate(axes[0].texts):
        t.set_text(t.get_text() + f"%   ({cm_model_1.ravel()[pos]})".format())

    sb.heatmap(np.reshape(cm_model_2_proc, (2,2)), annot=True, ax=axes[1], fmt='g', cmap=color_map)
    for pos, t in enumerate(axes[1].texts):
        t.set_text(t.get_text() + f"%   ({cm_model_2.ravel()[pos]})".format())

    fig.suptitle(fig_title, fontsize = 16)
    axes[0].set_title(model_1_title)
    axes[1].set_title(model_2_title)
    for axis in axes:
        axis.set_xlabel(x_label, fontsize = 14)
        axis.set_ylabel(y_label, fontsize = 14)    
        axis.yaxis.set_ticklabels([positive_label, negative_label])
        axis.xaxis.set_ticklabels([positive_label, negative_label])
        
actual = [0, 1, 0, 1, 1, 1]
model_1 = [1, 1, 1, 0, 1, 1]
model_2 = [0, 1, 0, 0, 1, 1]
confusion_matrix_comparison_plot(actual, model_1, model_2)
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import transforms
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import seaborn as sns
import matplotlib.style as style 
import matplotlib.pyplot as plt
style.use('ggplot')
# style.use('seaborn-poster') #sets the size of the charts

from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
import os
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, \
                            precision_score, recall_score, auc, roc_auc_score, \
                            confusion_matrix, roc_curve
from utils.print_pretty_confusion_matrix import plot_confusion_matrix_from_data 
sys.path.insert(0, 'datasets/')

from binary_data_loader import LAD_MPR_Loader

def estimate_model(y_true, y_pred, pred_probas):
    """
    Calculated all main metrics for binary classification:
        - accuracy, 
        - balanced_accuracy, 
        - precision, 
        - recall, 
        - f1, 
        - roc_auc_score
    Returns:
        - dict: dictionary, key - is the name of metric, value - it`s value=)  
    """
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, pred_probas)
    return {'accuracy': accuracy , 'balanced_accuracy': balanced_accuracy, 'precision': precision, 'recall': recall, 
            'f1': f1, 'roc_auc': roc_auc}

def show_estimate_model(calculated_metrics_dict, path_to_save=None, color_palette="muted"):
    """
    Plots barplot of the metrics of the model
    
    Args:
        - model_calculated_metrics(dict): dictionary, key - is the name of metric, value - it`s value=)  
    """
    
    # Make a fake dataset
    height = [calculated_metrics_dict['accuracy'], calculated_metrics_dict['balanced_accuracy'], 
              calculated_metrics_dict['precision'], calculated_metrics_dict['recall'], calculated_metrics_dict['f1'], 
              calculated_metrics_dict['roc_auc']]
    bars = ("Accuracy", "Balanced accuracy", "Precision", "Recall", "F1", "ROC/AUC score")
    y_pos = np.arange(len(bars))
    print
    plt.figure(figsize=(20,10))
    plt.bar(y_pos, height, color=sns.color_palette(color_palette))
    plt.xticks(y_pos, bars, rotation=0, fontsize=20)
    plt.ylim(0,1)
    plt.yticks(list(plt.yticks()[0]) + [0.5, 0.7, 0.9, 0.1, 0.3], fontsize=20)

    for i in range(len(y_pos)) :
        plt.text(i - 0.22, height[i]+0.03, str(round(height[i], 3)), color='#696969', fontweight='bold', fontsize=22)
    
    plt.ylabel('Metric\'s value', fontsize=30)
    plt.show()
    if path_to_save:
#         plt.savefig(os.path.join(path_to_save, 'metrics_of_the_model.png'))
          plt.savefig(path_to_save + '/metrics_of_the_model.png')

def plot_roc_curve(fpr, tpr, path_to_save=None):
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    if path_to_save:
        plt.savefig(os.path.join(path_to_save, 'roc_curve.png'))

def create_dataloader(path_to_csv, path_to_data, dataset_partition, batch_size):

    lad_dataset = LAD_MPR_Loader(path_to_csv, path_to_data, dataset_partition)
    lad_loader = torch.utils.data.DataLoader(lad_dataset,
                                                 batch_size=64, 
                                                 shuffle=False,
                                                )
    return lad_loader

def get_model_predictions(model, data_loader , device):
    all_preds = []
    all_labels = []
    all_probas = []
    all_patient_names = []
    all_img_names = []
    all_stenosis_scores = []
    model.eval()
    with torch.no_grad():
        for imgs, labels, stenosis_scores, patient_names, img_names in tqdm(data_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            softmax_logits = F.softmax(logits, dim=1)
            _,preds = torch.max(softmax_logits, 1)
            all_preds+= list(preds.cpu().numpy())
            all_labels+= list(labels.cpu().numpy())
            all_probas+= list(softmax_logits.detach().cpu().numpy())
            all_patient_names += list(patient_names)
            all_img_names += list(img_names)
            all_stenosis_scores += list(stenosis_scores)

    return all_preds, all_probas, all_labels, all_patient_names, all_img_names, all_stenosis_scores

def save_plots_all_info_per_dataset(model, data_loader, dataset_part, path_to_save_figures, device):
    if not os.path.exists(path_to_save_figures):
        os.mkdir(path_to_save_figures)

    # Get predictions
    all_preds, all_probas, all_labels, all_patient_names, all_img_names, all_stenosis_scores = get_model_predictions(model, data_loader, device)

    data = {'PATIENT_NAME': all_patient_names, 'IMG_NAME': all_img_names, 'PREDS': all_preds, 'PREDS_PROBAS': all_probas, 'LABELS': all_labels, 'STENOSIS_SCORE': all_stenosis_scores}
    df_to_save = pd.DataFrame(data)
    df_to_save.to_csv(os.path.join(path_to_save_figures, dataset_part+'.csv'))

    # Calculate all metrics
    fpr, tpr, thresholds = roc_curve(all_labels,  [x[1] for x in all_probas])
    calculated_metrics_dict = estimate_model(all_labels, all_preds, [x[1] for x in all_probas])

    # Plot with all metrics(Accuracy, F1 score, Precision, Recall, ROC/AUC score) 
    show_estimate_model(calculated_metrics_dict, path_to_save_figures)

    # Plot confusion matrics
    plot_confusion_matrix_from_data(all_labels, all_preds, columns=['Normal', 'Stenosis' ],
                               path_to_save=path_to_save_figures,
                               annot=True, cmap="Oranges",
                               fmt='.2f', fz=11, lw=0.5, cbar=False, 
                               figsize=[5,5], show_null_values=0, pred_val_axis='lin')
    # Plot ROC curve
    plot_roc_curve(fpr, tpr, path_to_save_figures)

if __name__ == '__main__':
    PATH_TO_MODEL_WEIGHTS = '/home/petryshak/CoronaryArteryPlaqueIdentification/weights/pretrained_resnet18_balanced_data.pth'

    PATH_TO_DATASET = 'data/binary_classification_only_lad'

    model_name = PATH_TO_MODEL_WEIGHTS.split('/')[-1].strip('.pth')
    PATH_TO_SAVE_GRAPHS = os.path.join('prediction_results/', model_name)

    if not os.path.exists(PATH_TO_SAVE_GRAPHS):
        os.mkdir(PATH_TO_SAVE_GRAPHS)

    # Create model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    # model = models.resnet50(pretrained=True)

    # model = models.resnext50_32x4d(pretrained=True, progress=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(PATH_TO_MODEL_WEIGHTS))
    model.to(device)

    # Predict over all datasets and visualize model's metrics
    datasets_partitions = ['train', 'val', 'test']

    csv_files = ['train.csv', 'val.csv', 'test.csv']
    # csv_files = ['train_without_25.csv', 'val_without_25.csv', 'test_without_25.csv']
    # csv_files = ['train_without_25_text_removed.csv', 'val_without_25_text_removed.csv', 'test_without_25_text_removed.csv']

    for i in tqdm(range(len(csv_files))):
        lad_loader = create_dataloader(os.path.join(PATH_TO_DATASET, csv_files[i]),
                                       PATH_TO_DATASET,
                                       datasets_partitions[i],
                                       1)

        save_plots_all_info_per_dataset(model, lad_loader, datasets_partitions[i], os.path.join(PATH_TO_SAVE_GRAPHS, datasets_partitions[i]), device)
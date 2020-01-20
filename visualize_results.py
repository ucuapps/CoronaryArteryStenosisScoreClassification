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

# from binary_data_loader import LAD_MPR_Loader
from estimate_model import *
from os.path import join
import re
from os import listdir
import cv2


def calculate_metrics(col_section, col_ids, col_preds, col_labels):
    """
    Calculate final auc and f1 metrics on three levels: per patient, per section and per artery
    :return: {dict} each metric as a key and its calculated metric as a value
    """
    assert len(col_section) == len(col_ids) == len(col_preds) == len(col_labels)

    metrics = {'ACC_section': 0, 'ACC_patient': 0, 'ACC_artery': 0, 'F1_section': 0, 'F1_patient': 0, 'F1_artery': 0}
    dict_artery = {'LAD': ['D-1', 'D-2', 'LAD', 'D-3', '2D-2', 'D-1Original', 'LADOriginal', 'D-4'],
                   'LCX': ['LCX', 'OM-2', 'OM-1', 'OM-3', 'OM', 'LCX-PLB', 'LCX-PDA', 'PLV_LCX', 'PDA_LCX'],
                   'RCA': ['RCA', 'RCA-PLB', 'RCA-PDA', 'PLV_RCA']}

    df = pd.concat([col_ids, col_section, col_preds, col_labels], axis=1)
    df = df.rename(columns={col_section.name: 'section', col_ids.name: 'patient', col_preds.name:
        'preds', col_labels.name: 'labels'})
    df['artery'] = df['section'].apply(lambda x: [k for k in dict_artery.keys() if x in dict_artery[k]][0])

    # SECTION
    section_labels = df[['preds', 'labels', 'section', 'artery', 'patient']].groupby(['patient', 'section']).agg(
        lambda x: max(x))
    preds_section = df[['preds', 'labels', 'section', 'artery', 'patient']].groupby(['patient', 'section']).agg(
        lambda x: x.value_counts().index[0])
    acc = accuracy_score(preds_section['preds'], section_labels['labels'])
    f1 = f1_score(preds_section['preds'], section_labels['labels'], average='weighted')
    metrics['ACC_section'], metrics['F1_section'] = acc, f1

    # ARTERY
    sect = section_labels.reset_index()
    artery_labels = sect.groupby(['patient', 'artery']).agg(lambda x: max(x))['labels']
    preds_artery = preds_section.reset_index().groupby(['patient', 'artery']).agg(lambda x: max(x))[
        'preds']  # x.value_counts().index[0])['preds']
    acc = accuracy_score(preds_artery, artery_labels)
    f1 = f1_score(preds_artery, artery_labels, average='weighted')
    metrics['ACC_artery'], metrics['F1_artery'] = acc, f1

    # PATIENT
    art = artery_labels.reset_index()
    patient_labels = art.groupby(['patient']).agg(lambda x: max(x))['labels']
    #     print(preds_artery.reset_index())
    preds_patient = preds_artery.reset_index().groupby(['patient']).agg(lambda x: max(x))[
        'preds']  # x.value_counts().index[0])['preds']
    acc = accuracy_score(preds_patient, patient_labels)
    f1 = f1_score(preds_patient, patient_labels, average='weighted')
    metrics['ACC_patient'], metrics['F1_patient'] = acc, f1

    return metrics


def label_predictions_to_images(partition, path_model, df, type_pred, p_data):
    """Draw predictions and labels on images and saves to trained model's folder"""
    
    folder_imag = path_model + '/images'
    folder_imag_correct = path_model + '/images/correct'
    folder_imag_mistakes = path_model + '/images/mistakes'
    if not os.path.exists(folder_imag):
        os.mkdir(folder_imag)
    if not os.path.exists(folder_imag_correct):
        os.mkdir(folder_imag_correct)
    if not os.path.exists(folder_imag_mistakes):
        os.mkdir(folder_imag_mistakes)

    for index,a in df.iterrows():
        im_name = [x for x in listdir(join(join(p_data, a['PATIENT_NAME']), a['SECTION'])) if a['IMG_NAME'] in x][0]
        img = cv2.imread(join(p_data,a['PATIENT_NAME'],a['SECTION'],im_name),0)
        label_text = 'Label: ' + str(a['LABELS']) + '  stenosis: ' + a['STENOSIS'] + '%'
        pred_text = '    |   Prediction: ' + str(a['PREDS']) + '  prob: ' + str(round(a['PROBABILITY']*100, 3))+ '%'

        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(img,label_text + pred_text,(10,500), font, 0.4,(255,255,255),1,cv2.LINE_AA)
        if type_pred == 'correct':
            cv2.imwrite( join(folder_imag_correct, im_name), img )
        if type_pred == 'mistake':
            cv2.imwrite( join(folder_imag_mistakes, im_name), img )
    return 


def categorize_stenosis_score(stenosis_scores):
    temp = [' '.join(re.sub('[>,%<]', ' ', el).split('___')).replace('NORMAL', '0').split() for el in stenosis_scores]
    d = {'0':1, '25':2, '25-50':3, '50':4,'50-70':5, '70':6}
    temp = [max([d[i] for i in t ]) for t in temp]
    stenosis_scores = [list(d.keys())[list(d.values()).index(el)] for el in temp]
    return stenosis_scores


def show_bars(list_values, list_names, size_sections, name_metric, color_palette="muted", path_to_save=None, save_fig=False, plot_mode=None):
    y_pos = np.arange(len(list_names))
    plt.figure(figsize=(12,6))
    plt.bar(y_pos, list_values, color=sns.color_palette(color_palette))
    plt.xticks(y_pos, list_names, rotation=0, fontsize=20)
    plt.ylim(0,1)
    plt.yticks(list(plt.yticks()[0]) + [0.5, 0.7, 0.9, 0.1, 0.3], fontsize=20)

    for i in range(len(y_pos)) :
        plt.text(i - 0.22, list_values[i]+0.03, str(round(list_values[i], 3)) +' (' + str(size_sections[i])+ ')', color='#696969', fontweight='bold', fontsize=14)

    plt.ylabel(name_metric, fontsize=30)
    if not save_fig:
        plt.show()
    else:
        plt.savefig(os.path.join(path_to_save, '{}_by_{}.png'.format(name_metric, plot_mode)))
        
    
def plot_accuracy_bystage(preds_df, path_to_save_figures):
    list_accuracies, size_stages = [], []
    list_stages = ['0', '25', '25-50', '50', '50-70', '70']

    for stenosis_stage in list_stages:
        stage_df = preds_df[preds_df['STENOSIS']==stenosis_stage]

        labels_stage = stage_df['LABELS'].values
        preds_stage = stage_df['PREDS'].values
        probas_stage = stage_df['PREDS_PROBAS'].values
        new_probas_stage = [[float(el) for el in arr[1:-1].split(' ') if len(el)>2] for arr in probas_stage]

        accuracy = accuracy_score(labels_stage, preds_stage)
        list_accuracies.append(accuracy), size_stages.append(len(stage_df))
    show_bars(list_accuracies, list_stages, size_stages, 'Accuracy',path_to_save=path_to_save_figures, save_fig=True,plot_mode='stage')
    
    
def plot_accuracy_bysection(preds_df, path_to_save_figures):
    list_accuracies,list_f1,list_recall, list_precision, size_sec = [], [], [], [], []
    list_sections = preds_df['IMG_NAME'].apply(lambda x: x.split('_')[0]).unique()

    for section in list_sections:
        sec_df = preds_df[preds_df['IMG_NAME'].str.contains(section)]

        labels_sec = sec_df['LABELS'].values
        preds_sec = sec_df['PREDS'].values
        probas_sec = sec_df['PREDS_PROBAS'].values
        new_probas_sec = [[float(el) for el in arr[1:-1].split(' ') if len(el)>2] for arr in probas_sec]

        accuracy, f1 = accuracy_score(labels_sec, preds_sec), f1_score(labels_sec, preds_sec)
        recall, precision = recall_score(labels_sec, preds_sec), precision_score(labels_sec, preds_sec)
        list_accuracies.append(accuracy), size_sec.append(len(sec_df)), list_f1.append(f1), list_recall.append(recall), list_precision.append(precision)

    for metric in ['Accuracy', 'F1 score', 'Recall', 'Precision']:
         show_bars(list_accuracies, list_sections, size_sec, metric, path_to_save=path_to_save_figures, save_fig=True, plot_mode='section')
            
            
def calc_auc_bypatient(preds_df):
    preds_df['PATIENT_ID'] = preds_df['IMG_NAME'].apply(lambda x: x.split('_')[2])
    preds_df['SECTION'] = preds_df['IMG_NAME'].apply(lambda x: x.split('_')[0])
    patients_ids = preds_df['PATIENT_ID'].unique()
    sections_ids = preds_df['SECTION'].unique()
    patients_auc = {}
    patients_df = preds_df[['LABELS', 'PREDS', 'PATIENT_ID', 'SECTION']].groupby(['PATIENT_ID', 'SECTION'], as_index=False).count()[['PATIENT_ID', 'SECTION']]
    patients_df['AUC'] = None

    for p in patients_ids:
        sections_dict = {}
        for s in sections_ids:
            temp_df = preds_df[preds_df['PATIENT_ID'] == p][preds_df['SECTION'] == s]
            temp_pat = patients_df[patients_df['PATIENT_ID']==p][patients_df['SECTION']==s]
            if len(temp_df) > 0:
                acc_patient_section = accuracy_score(temp_df['LABELS'], temp_df['PREDS'])
                sections_dict[s] = acc_patient_section
                patients_df.xs(temp_pat.index[0])['AUC'] = float(acc_patient_section)
        patients_auc[p] = sections_dict
    patients_df['AUC'] = patients_df['AUC'].astype(float)
    auc_bypatient = preds_df[['LABELS', 'PREDS', 'PATIENT_ID', 'SECTION']].groupby(['PATIENT_ID', 'SECTION'], as_index=False).sum()
    return patients_df, auc_bypatient


if __name__ == '__main__':
    DATA_SPLIT = 'val'
    # DATA_SPLIT = 'test'
    MODEL_NAME = 'retrained_resnet18_balanced_data_without_25_text_removed_l2_regularization'
    SAVE_LABELED_IMAGES = True
    
    p = 'prediction_results/{}/{}/'.format(MODEL_NAME, DATA_SPLIT)
    p_data = 'data/binary_classification_only_lad/'
    data_df = pd.read_csv(os.path.join(p_data, 'test_without_25_text_removed.csv'))
    preds_df = pd.read_csv(os.path.join(p, '{}.csv'.format(DATA_SPLIT)))
    path_to_save_figures = join('/home/petryshak/CoronaryArteryPlaqueIdentification/prediction_results/',MODEL_NAME, DATA_SPLIT, 'visualization')
    if not os.path.exists(path_to_save_figures):
        os.mkdir(path_to_save_figures)
    
    labels = preds_df['LABELS'].values
    preds = preds_df['PREDS'].values
    probas = preds_df['PREDS_PROBAS'].values
    
    preds_df['PATIENT_ID'] = preds_df['IMG_NAME'].apply(lambda x: x.split('_')[2])
    preds_df['SECTION'] = preds_df['IMG_NAME'].apply(lambda x: x.split('_')[0])
    preds_df['STENOSIS'] = categorize_stenosis_score(preds_df['STENOSIS_SCORE'].values)
    all_probas = [[float(el) for el in arr[1:-1].split(' ') if len(el)>2] for arr in probas]
    preds_df['PROBABILITY'] = [x[1] for x in all_probas]
    
    patients_ids = preds_df['PATIENT_ID'].unique()
    sections_ids = preds_df['SECTION'].unique()
    
    # Draw graphs
    calculated_metrics_dict = estimate_model(labels, preds, [x[1] for x in all_probas])
    show_estimate_model(calculated_metrics_dict, path_to_save_figures)
    
    plot_confusion_matrix_from_data(labels, preds, columns=['Normal', 'Stenosis' ], path_to_save=path_to_save_figures, annot=True, cmap='binary', fz=20, figsize=[9,9], show_null_values=0)
    
    fpr, tpr, thresholds = roc_curve(labels,  [x[1] for x in all_probas ])  
    plot_roc_curve(fpr, tpr, path_to_save_figures)
    
    plot_accuracy_bystage(preds_df, path_to_save_figures)
    plot_accuracy_bysection(preds_df, path_to_save_figures)
    
    patients_df, auc_bypatient = calc_auc_bypatient(preds_df)
    auc_bypatient.to_csv(join(path_to_save_figures, 'auc_by_patient_section.csv'), index=False)
    
    if SAVE_LABELED_IMAGES:
        # Save images by section
        corrects = preds_df[preds_df['LABELS'] == preds_df['PREDS']]
        corrects = corrects.groupby(['SECTION', 'LABELS']).head(10)  # select 10 examples from each group
        label_predictions_to_images(DATA_SPLIT, path_to_save_figures, corrects, 'correct')

        # Save all images where mistakes have been made 
        mistakes = preds_df[preds_df['LABELS'] != preds_df['PREDS']]
        label_predictions_to_images(DATA_SPLIT, path_to_save_figures, mistakes, 'mistake')

    print("'\n'Completed!'\n'")

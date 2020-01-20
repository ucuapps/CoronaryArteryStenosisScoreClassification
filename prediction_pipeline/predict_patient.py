import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from patient_data_structure import PatientDataStructure
from pred_datasets import MPR_Dataset
from pred_models import ShuffleNetv2

class PatientPredictor(object):

    """
    Class for prediction the patient's MPR data. 

    Args:
        path_to_weights: path to the weights of the best model(best_model_shufflenetv2.pth)
        device: device which use for neural network prediction

    Example:
        PATH = 'prediction_pipeline/DICOMOBJ/'
        PATH_TO_WEIGHTS = 'best_model_shufflenetv2.pth'
        
        # create the instance of the class
        patient_predictor = PatientPredictor(PATH_TO_WEIGHTS, device='cuda')

        # get the predictions for each branch of the coronary artery for the given patient
        patient_predictor.predict(PATH_TO_THE_PATIENT_DICOM_FILES)
    """
    def __init__(self, path_to_weighs, device='cpu'):

        # ToDo parametrize model
        self.classes_enoding = {0: 'Normal', 1: 'Insignificant stenosis',2: 'Significant stenosis'}
        self.device = device
        self.model = ShuffleNetv2(n_classes=3)
        self.model.load_state_dict(torch.load(path_to_weighs,map_location={'cuda:0': device}))
        self.model.to(device)


    def predict(self, path_to_patient_dcms):
        """
        Predicts the stenosis score for the given patient per branch

        Args:
            path_to_patient_dcms: path to the patient dicom files

        Returns:
            dict: key(name of the artery branch) - value(predicted class for that branch)
        """
        loader = self.create_patient_loader(path_to_patient_dcms)
        predictions = {}

        for imgs, artery in loader:
            imgs = torch.squeeze(imgs).to(self.device)
            artery = artery[0]
            output = self.model(imgs).cpu()
            softmax_output = F.softmax(output, dim=1)
            _, pred_classes = softmax_output.max(1)
            pred_classes = pred_classes.numpy()

            # Major voting
            (class_values,class_frequency) = np.unique(pred_classes,return_counts=True)
            ind_of_max_class = np.argmax(class_frequency)
            predictions[artery] = class_values[ind_of_max_class]
        return predictions


    def create_patient_loader(self, path_to_patient_dcms):
        patient_dict = PatientDataStructure(path_to_patient_dcms).dict_dataset

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        # ToDo parametrize dataloader to be able to run with report file
        dataloader = DataLoader(MPR_Dataset(patient_dict, transform=transform), shuffle=False, batch_size=1)
        return dataloader


if __name__ == '__main__':
    PATH = '/home/petryshak/CoronaryArteryPlaqueIdentification/prediction_pipeline/DICOMOBJ/'
    PATH_TO_WEIGHTS = '../model_model_34_val_f1=0.9360136.pth'

    patient_predictor = PatientPredictor(PATH_TO_WEIGHTS, device='cuda')

    a = patient_predictor.predict(PATH)
    print(a)
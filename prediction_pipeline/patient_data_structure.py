import pandas as pd
import pydicom as dicom
import numpy as np
import os

class PatientDataStructure(object):

    def __init__(self, path_to_data):
        self.path_to_data = path_to_data
        self.dict_dataset = self.all_arteries()

    def artery(self, artery_name):
        return self.dict_dataset[artery_name]

    def all_arteries(self):
        patient_dictionary = self.__get_patient_dictionary()
        splited_mpr_names_filtered = [self.__map_mpr_name_to_record_name(x) for x in patient_dictionary]
        dict_keys = list(patient_dictionary.keys())

        # change keys in the dict to the corresponding labels in the reports
        for k in range(len(dict_keys)):
            if splited_mpr_names_filtered[k]:
                if dict_keys[k] in splited_mpr_names_filtered:
                    pass
                else:
                    patient_dictionary[splited_mpr_names_filtered[k]] = patient_dictionary[dict_keys[k]]
                    del patient_dictionary[dict_keys[k]]
            else:
                del patient_dictionary[dict_keys[k]]

        if 'THRASH' in patient_dictionary.keys():
            del patient_dictionary['THRASH']

        return patient_dictionary


    def __get_patient_dictionary(self):
        """
        Returns dict of different types of images in the folder of patient. 

        Returns:
            dict: key - type of images; value - path_to_corresponding DICOM file.
        """
        patient_dict = {}
        
        dicom_file_names = os.listdir(self.path_to_data)
        
        for i in range(len(dicom_file_names)):
            cur_dicom_obj = dicom.dcmread(os.path.join(self.path_to_data, dicom_file_names[i]))
            
            if cur_dicom_obj.SeriesDescription not in patient_dict.keys():
                patient_dict[cur_dicom_obj.SeriesDescription] = []
            patient_dict[cur_dicom_obj.SeriesDescription].append(os.path.join(self.path_to_data, dicom_file_names[i]))
        # sort each type of images with respect to their depth in ascending order
    #     for i in patient_dict:
    #         patient_dict[i].sort(key=lambda x: x.InstanceNumber)

        return patient_dict

    def __map_mpr_name_to_record_name(self, mpr_name):
        # ToDo handle PLB PLV branches
        main_branches_dict = {
            'LAD': ['LAD', 'LAD ', 'LAD Original', 'LAD original', 'LAD *', 'LAD*'],
            'D-1':['LAD-D1 original', 'LAD-D1 Original', 'LAD-D1', 'LAD-D1 *', 'LAD -D1', 'LAD -D1', 'LAD - D1', 'D1'],
            'D-2':['LAD-D2', 'LAD-D2 *', 'LAD-D2', '2LAD-D2', 'LAD -D2', 'LAD-D2 original', 'LAD -D2'],
            'D-3': ['LAD-D3', 'LAD-D3 *', 'LAD-D3', 'LAD-D3 original', ],
            'D-4': [ 'LAD - D4 *', 'LAD-D4', 'LAD-D4 *'],
            'RCA': ['RCA', 'RCA *', 'RCA*', 'RCA original'],
            'OM':['OM*', 'LCX-OM  *', 'OM *', 'OM', 'LCX-OM*', 'LCX - OM *', 'LCX-OM original', 'LCX-OM *', 'LCX-OM', 'OM original'],
            'OM-1': ['LCX-OM1 *', 'OM1 *', 'OM1', 'LCX-OM1', 'LCX -OM1 *', 'LCX-OM1*'],
            'OM-2': ['LCX-OM2 *', 'OM2 *', 'LCX-OM2', 'LCX - OM2 *', 'LCX -OM2 *', 'OM2*', 'LCX-OM2*'],
            'OM-3': ['LCX-OM3 *', 'LCX -OM3 *', 'OM3',  'LCX-OM3*', 'LCX-OM3', 'OM3 *', 'OM3*'],
            'OM-4': ['OM4 *', 'OM4', 'LCX-OM4 *'],
            'LCX': ['LCX', 'LCX *', 'LCX original', 'LCX  *', 'LCX*'],
            'PDA_RCA': ['RCA-PDA','RCA -PDA', 'RCA-PDA*', 'RCA-PDA *', 'RCA-PDA1','RCA-PDA2', 'RCA-PDA2 *','RCA-PDA2', 
                        'RCA-PDA2*'],
            'PLV_RCA': ['RCA-PLB', 'RCA-PLB ', 'RCA-PLB', 'RCA -PLB*', 'RCA-PLB1 *','RCA-PLB1', 'RCA-PLB1 *','RCA-PLB2', 
                        'RCA-PLB2 *'],
            'PDA_LCX': ['LCX-PDA *', 'LCX-PDA', 'LCX-PDA2', 'LCX-PDA2 *'],
            'PLV_LCX': ['LCX-PLB', 'LCX-PLB *', 'LCX-PLB1', 'LCX-PLB2',  'LCX-PLB2 *'],
            'THRASH': ['PLB  *', 'PLB *', 'PLB original', 'PLB','PLB*','PLB1 *', 'PLB1*', 'PLB1','PLB2 *', 'PLB2*', 'PLB2']
        }
        
        for key in main_branches_dict:
            if mpr_name in main_branches_dict[key]:
                return key

    def __split_mpr_name(self, mpr_name):
        return \
            "".join(mpr_name.split()).replace('*', '').replace('original', '') \
            .replace('LIMA-', '').replace('Branchof','').replace('TOPDA', '').replace('PDATO', '')

if __name__ == '__main__':
    patient_data = PatientDataStructure('/home/petryshak/CoronaryArteryPlaqueIdentification/prediction_pipeline/DICOMOBJ/')
    print(patient_data.dict_dataset.keys())
#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

'''
This function aims to copy raw data from source to the destination and generate jason file
'''

from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    base = "D:/2023 code/nnUNet-master/dataset/uuUNet_raw"
    amos_base_dir = 'G:/YangLab_FANN-main/data/EXACT09'
    test_patient_names = ['Airway_0292', 'Airway_0328', 'Airway_0371', 'Airway_0403',
                           'Airway_0430', 'Airway_0488', 'Airway_0657', 'Airway_0696',
                           'Airway_0710', 'Airway_0728', 'Airway_0757', 'Airway_0806',
                           'Airway_0819', 'Airway_0909', 'Airway_0981', 'Airway_CASE02', 
                           'Airway_CASE14', 'Airway_CASE17']
    fids_train_val = []
    p = join(amos_base_dir, 'LIDC')
    for fid in os.listdir(p):
        if fid[:-12] not in test_patient_names:
            fids_train_val.append(fid[:-12])
    train_patient_names, valid_patient_names = train_test_split(fids_train_val, test_size=0.2, random_state=0)

    # Jason generation
    task_id = 1
    task_name = "BAS"
    prefix = 'Airway'

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join('D:/2023 code/nnUNet-master/dataset/nnUNet_raw', foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    # copy raw data
    for tr in train_patient_names:
        shutil.copy(join(amos_base_dir, 'LIDC', tr + '_0000.nii.gz'), join(imagestr, f'{tr}.nii.gz'))
        shutil.copy(join(amos_base_dir, 'LIDC70_gt', tr + '.nii.gz'), join(labelstr, f'{tr}.nii.gz'))

    for ts in test_patient_names:
        shutil.copy(join(amos_base_dir, 'LIDC', ts + '_0000.nii.gz'), join(imagests, f'{ts}.nii.gz'))

    for vl in valid_patient_names:
        shutil.copy(join(amos_base_dir, 'LIDC', vl + '_0000.nii.gz'), join(imagestr, f'{vl}.nii.gz'))
        shutil.copy(join(amos_base_dir, 'LIDC70_gt', vl + '.nii.gz'), join(labelstr, f'{vl}.nii.gz'))


    train_folder = join(base, "Training/img")
    label_folder = join(base, "Training/label")
    test_folder = join(base, "Test/img")

    json_dict = OrderedDict()
    json_dict['name'] = "BAS"
    json_dict['description'] = "Airway Segmentation"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = None
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
    }
    json_dict['labels'] = OrderedDict({
        "00": "background",
        "01": "Airway",
        }
    )
    json_dict['numTraining'] = len(fids_train_val)
    json_dict['numTest'] = len(test_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s" % train_patient_name, "label": "./labelsTr/%s" % train_patient_name} for i, train_patient_name in enumerate(train_patient_names)]
    json_dict['test'] = ["./imagesTs/%s" % test_patient_name for test_patient_name in test_patient_names]
    json_dict['validation'] = [{'image': "./imagesTr/%s" % valid_patient_name, "label": "./labelsTr/%s" % valid_patient_name} for i, valid_patient_name in enumerate(valid_patient_names)]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))


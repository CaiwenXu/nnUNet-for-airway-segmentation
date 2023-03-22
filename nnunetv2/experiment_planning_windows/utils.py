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

import json
import os
import pickle
import shutil
from collections import OrderedDict
from multiprocessing import Pool

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, isdir, subfiles, subdirs, isfile
from nnunetv2.configuration_windows import default_num_threads
from nnunetv2.experiment_planning_windows.common_utils import split_4d_nifti
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities_windows.file_endings import get_last_folder


def split_4d(input_folder, num_processes=default_num_threads, overwrite_task_output_id=None):
    assert isdir(join(input_folder, "imagesTr")) and isdir(join(input_folder, "labelsTr")) and \
           isfile(join(input_folder, "dataset.json")), \
        "The input folder must be a valid Task folder from the Medical Segmentation Decathlon with at least the " \
        "imagesTr and labelsTr subfolders and the dataset.json file"

    full_task_name = get_last_folder(input_folder)

    assert full_task_name.startswith("Task"), "The input folder must point to a folder that starts with TaskXX_"

    first_underscore = full_task_name.find("_")
    assert first_underscore == 6, "Input folder start with TaskXX with XX being a 3-digit id: 00, 01, 02 etc"

    input_task_id = int(full_task_name[4:6])
    if overwrite_task_output_id is None:
        overwrite_task_output_id = input_task_id

    task_name = full_task_name[7:]

    output_folder = join(nnUNet_raw, "Task%03.0d_" % overwrite_task_output_id + task_name)

    if isdir(output_folder):
        shutil.rmtree(output_folder)

    files = []
    output_dirs = []

    os.makedirs(output_folder, exist_ok=True)
    if not isdir(output_folder):
        os.mkdir(output_folder)

    for subdir in ["imagesTr", "imagesTs"]:
        curr_out_dir = join(output_folder, subdir)
        if not isdir(curr_out_dir):
            os.mkdir(curr_out_dir)
        curr_dir = join(input_folder, subdir)
        nii_files = [join(curr_dir, i) for i in os.listdir(curr_dir) if i.endswith(".nii.gz")]
        nii_files.sort()
        for n in nii_files:
            files.append(n)
            output_dirs.append(curr_out_dir)

    shutil.copytree(join(input_folder, "labelsTr"), join(output_folder, "labelsTr"))

    p = Pool(num_processes)
    p.starmap(split_4d_nifti, zip(files, output_dirs))
    p.close()
    p.join()
    shutil.copy(join(input_folder, "dataset.json"), output_folder)



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
import sys
sys.path.append(".")
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.configuration_windows import default_num_threads
from nnunetv2.experiment_planning_windows.utils import split_4d
from nnunetv2.utilities_windows.file_endings import remove_trailing_slash


def crawl_and_remove_hidden_from_decathlon(folder):
    folder = remove_trailing_slash(folder)
    subf = subfolders(folder, join=False)

    valid_folder = os.path.basename(folder).startswith("Task")
    valid_folder &= ('imagesTr' in subf) and ('imagesTs' in subf) and ('labelsTr' in subf)
    assert valid_folder, "This does not seem to be a decathlon folder. Please give me a folder that starts with " \
                         "TaskXX and has the subfolders 'imagesTr', 'labelsTr' and 'imagesTs'."

    _ = [os.remove(i) for i in subfiles(folder, prefix=".")]
    _ = [os.remove(i) for i in subfiles(join(folder, 'imagesTr'), prefix=".")]
    _ = [os.remove(i) for i in subfiles(join(folder, 'labelsTr'), prefix=".")]
    _ = [os.remove(i) for i in subfiles(join(folder, 'imagesTs'), prefix=".")]


def main():
    import argparse
    from nnunetv2.script_util import (
    args_to_dict,
    add_dict_to_argparser,
)
    # parser = argparse.ArgumentParser(description="The MSD provides data as 4D Niftis with the modality being the first"
    #                                              " dimension. We think this may be cumbersome for some users and "
    #                                              "therefore expect 3D niftixs instead, with one file per modality. "
    #                                              "This utility will convert 4D MSD data into the format nnU-Net "
    #                                              "expects")
    # parser.add_argument("-i", help="Input folder. Must point to a TaskXX_TASKNAME folder as downloaded from the MSD "
    #                                "website", required=True)
    # parser.add_argument("-p", required=False, default=default_num_threads, type=int,
    #                     help="Use this to specify how many processes are used to run the script. "
    #                          "Default is %d" % default_num_threads)
    # parser.add_argument("-output_task_id", required=False, default=None, type=int,
    #                     help="If specified, this will overwrite the task id in the output folder. If unspecified, the "
    #                          "task id of the input folder will be used.")
    # args = parser.parse_args()

    def create_argparser():
        defaults = dict(
            i="D:/2023 code/nnUNet-master/dataset/nnUNet_raw/Task01_BAS",
            # data_dir="/jmain02/home/J2AD015/axf04/xxx16-axf04/caiwen/ATM22_cut_1_64_64/train/",
            p=default_num_threads,
            output_task_id=None
        )
        parser = argparse.ArgumentParser()
        add_dict_to_argparser(parser, defaults)
        return parser
    
    args = create_argparser().parse_args()
    crawl_and_remove_hidden_from_decathlon(args.i)

    split_4d(args.i, args.p, args.output_task_id)


if __name__ == "__main__":
    main()

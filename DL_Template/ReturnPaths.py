__author__ = 'Brian M Anderson'
# Created on 2/17/2020

import os
from Deep_Learning.Base_Deeplearning_Code.Data_Generators.Return_Paths import find_base_dir, find_raid_dir


def __init__():
    pass


def return_paths():
    pass
    try:
        base = r'\\mymdafiles\di_data1'
        base_path = r'H:\Deeplearning_Recurrence_Work\Nifti_Exports\Records'
        os.listdir(base_path)
        morfeus_drive = os.path.abspath(
            os.path.join(base, 'Morfeus', 'BMAnderson', 'Modular_Projects'))
    except:
        base = find_raid_dir()
        base_path = os.path.join(base)
        morfeus_drive = os.path.abspath(
            os.path.join(find_base_dir(), 'Morfeus', 'BMAnderson', 'Modular_Projects'))

    return base_path, morfeus_drive


if __name__ == '__main__':
    pass

__author__ = 'Brian M Anderson'
# Created on 4/30/2021
from Base_Deeplearning_Code.Dicom_RT_and_Images_to_Mask.src.DicomRTTool import DicomReaderWriter, plot_scroll_Image


def identify_rois_in_path(path):
    reader = DicomReaderWriter()
    reader.which_indexes_have_all_rois()
    reader.walk_through_folders(input_path=path)
    rois = reader.return_rois(print_rois=True)
    return reader, rois


if __name__ == '__main__':
    pass

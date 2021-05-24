__author__ = 'Brian M Anderson'
# Created on 1/8/2021
from Base_Deeplearning_Code.Dicom_RT_and_Images_to_Mask.src.DicomRTTool import DicomReaderWriter


def dicom_to_nifi(nifti_path, dicom_path, Contour_Names, associations, excel_file, description):
    """
    Args:
        nifti_path: path to write out nifti files
        dicom_path: path to DICOM folders
        Contour_Names: names of desired rois
        associations: associations for rois
        excel_file: an excel file path to write out the data
        description: a description of the data

    Returns:

    """
    reader = DicomReaderWriter(Contour_Names=Contour_Names, associations=associations, description=description)
    reader.walk_through_folders(dicom_path)
    reader.write_parallel(out_path=nifti_path, excel_file=excel_file)
    return None


if __name__ == '__main__':
    pass

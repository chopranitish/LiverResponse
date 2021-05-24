__author__ = 'Brian M Anderson'
# Created on 9/28/2020
import os


'''
All of the top part will be in PreProcessingTools
'''

"""
First, identify the rois present in the scan
"""
associations = {'liver_bma_program_4': 'liver'}
contour_names = ['liver_segment_{}_bmaprogram3'.format(i) for i in range(1, 5)]
contour_names += ['liver_segment_5-8_bmaprogram3', 'liver']
dicom_path = r'C:\Morfeus_Lab\Liver'
identify_rois = True
if identify_rois:
    from PreProcessingTools.Dicom_Tools.Identify_Rois import identify_rois_in_path
    reader, rois = identify_rois_in_path(path=dicom_path)
    reader.set_contour_names_and_associations(Contour_Names=contour_names, associations=associations)
    reader.which_indexes_lack_all_rois()  # Check and see if there are indexes that lack the rois

"""
Next, write the images and masks to nifti files
"""

write_nifti = True
if write_nifti:
    '''
    This will print if any rois are missing at certain locations
    '''
    check_rois = True
    from Base_Deeplearning_Code.Dicom_RT_and_Images_to_Mask.src.DicomRTTool import DicomReaderWriter, plot_scroll_Image
    import os
    import numpy as np


    # contour_names += ['liver_segment_{}_bmaprogram4'.format(i) for i in range(1, 5)]
    # contour_names += ['liver_segment_5-8_bmaprogram4']
    for MRN in os.listdir(dicom_path):
        patient_path = os.path.join(dicom_path, MRN)
        primary_reader = DicomReaderWriter(description='Liver_Outcome_Primary', arg_max=False, get_dose_output=True)
        primary_reader.set_contour_names_and_associations(Contour_Names=contour_names, associations=associations)
        secondary_reader = DicomReaderWriter(description='Liver_Outcome_Followup', arg_max=False)
        secondary_reader.set_contour_names_and_associations(Contour_Names=contour_names, associations=associations)
        primary_reader.walk_through_folders(os.path.join(patient_path, 'Primary'))
        secondary_reader.walk_through_folders(os.path.join(patient_path, 'Secondary'))

        primary_reader.write_parallel(out_path=dicom_path, excel_file=os.path.join(dicom_path, 'MRN_Path_To_Iteration.xlsx'))
        secondary_reader.write_parallel(out_path=dicom_path, excel_file=os.path.join(dicom_path, 'MRN_Path_To_Iteration.xlsx'),thread_count=5)

resample_dose = False
if resample_dose:
    import SimpleITK as sitk
    import os
    path = r'C:\Morfeus_Lab\Liver'
    dose_files = [i for i in os.listdir(path) if i.startswith('Overall_dose')]
    for file in dose_files:
        print(file)
        primary_file = file.replace('dose', 'Data')
        dose_handle = sitk.ReadImage(os.path.join(path, file))
        primary_handle = sitk.ReadImage(os.path.join(path, primary_file))
        if dose_handle.GetSize() != primary_handle.GetSize():
            # print('These are not the same for {}...'.format(MRN))
            dose_handle = sitk.Resample(dose_handle, primary_handle, sitk.AffineTransform(3), sitk.sitkLinear,
                                        0, dose_handle.GetPixelID())
            sitk.WriteImage(dose_handle, os.path.join(path, file))

'''
Ensure that all contours are within the liver contour, as sometimes they're drawn to extend past it
'''

write_records = False
if write_records:
    from Base_Deeplearning_Code.Image_Processors_Module.src.Processors import MakeTFRecordProcessors as Processors
    from Base_Deeplearning_Code.Image_Processors_Module.src.Processors import TFRecordWriter as Writer
    import os
    image_path = os.path.join(nifti_path, 'Overall_Data_Test_0.nii.gz')
    annotation_path = os.path.join(nifti_path, 'Overall_mask_Test_y0.nii.gz')
    processors = [
        Processors.LoadNifti(nifti_path_keys=('image_path', 'annotation_path'),
                             # Loads a file path as a SimpleITK Image
                             out_keys=('image_handle', 'annotation_handle')),
        Processors.SimpleITKImageToArray(nifti_keys=('image_handle', 'annotation_handle'),  # Converts an Image to array
                                         out_keys=('image_array', 'annotation_array'), dtypes=('float32', 'int8'))
    ]
    normalizing_processors = [
        Processors.Threshold_Images(image_keys=('image_array',), lower_bound=-100, upper_bound=200),
        Processors.Box_Images(image_keys=('image_array',), annotation_key='annotation_array', wanted_vals_for_bbox=(1,),
                              bounding_box_expansion=(0, 0, 0), power_val_z=1, power_val_r=512,
                              power_val_c=512, min_images=None, min_rows=None, min_cols=None)
    ]
    distributing_processors = [
        Processors.DistributeInTo2DSlices(image_keys=('image_array', 'annotation_array'))
    ]


    record_path = r'H:\TF_Record_Exports'
    record_writer = Writer.RecordWriter(out_path=record_path, file_name_key='out_name', rewrite=True)
    file_dictionary_list = []
    files = [i for i in os.listdir(nifti_path) if i.startswith('Overall_Data')]
    for file in files:
        index = file.split('_')[-1].split('.nii')[0]
        image_path = os.path.join(nifti_path, file)
        annotation_path = os.path.join(nifti_path, 'Overall_mask_Test_y{}.nii.gz'.format(index))
        temp_dict = {'image_path': image_path, 'annotation_path': annotation_path,
                     'out_name': '{}.tfrecord'.format(index)}
        file_dictionary_list.append(temp_dict)
    Writer.parallel_record_writer(dictionary_list=file_dictionary_list, max_records=1,
                                  image_processors=processors + normalizing_processors + distributing_processors,
                                  recordwriter=record_writer, verbose=True, debug=True)
    # from Local_Recurrence_Work.Outcome_Analysis.PreProcessingTools.Nifti_to_tfrecords import nifti_to_records, os
    # nifti_to_records(nifti_path=os.path.join(nifti_export_path, 'Train'))
    # nifti_to_records(nifti_path=os.path.join(nifti_export_path, 'Validation'))
    # nifti_to_records(nifti_path=os.path.join(nifti_export_path, 'Test'))

print("All finished here, now move on to MainDeepLearning.py!")

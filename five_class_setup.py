from os import listdir
from os.path import isfile, join
from os.path import exists
import pandas as pd
import os


def five_class_image_text_label(dir_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/"):

    # negative_dir = 'Z:/Lymphoma_UW_Retrospective/Data/mips/Group_1_2_3_curated'
    # negative_dir = '/home/zmh001/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Data/mips/Group_1_2_3_curated'
    # positive_dir = 'Z:/Lymphoma_UW_Retrospective/Data/mips/Group_4_5_curated'
    # positive_dir = '/home/zmh001/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Data/mips/Group_4_5_curated'

    negative_dir = os.path.join(dir_base, 'Lymphoma_UW_Retrospective/Data/mips/Group_1_2_3_curated')
    positive_dir = os.path.join(dir_base, 'Lymphoma_UW_Retrospective/Data/mips/Group_4_5_curated')

    # gets all the file names in and puts them in a list
    neg_files = [f for f in listdir(negative_dir) if isfile(join(negative_dir, f))]
    pos_files = [f for f in listdir(positive_dir) if isfile(join(positive_dir, f))]

    if "Thumbs.db" in neg_files:
        neg_files.remove("Thumbs.db")
    if "Thumbs.db" in pos_files:
        pos_files.remove("Thumbs.db")

    all_files = neg_files + pos_files

    report_direct = os.path.join(dir_base, 'Lymphoma_UW_Retrospective/Reports')
    reports_1 = pd.read_csv(os.path.join(report_direct, 'ds1_findings_and_impressions_wo_ds_more_syn.csv'))
    reports_2 = pd.read_csv(os.path.join(report_direct, 'ds2_findings_and_impressions_wo_ds_more_syn.csv'))
    reports_3 = pd.read_csv(os.path.join(report_direct, 'ds3_findings_and_impressions_wo_ds_more_syn.csv'))
    reports_4 = pd.read_csv(os.path.join(report_direct, 'ds4_findings_and_impressions_wo_ds_more_syn.csv'))
    reports_5 = pd.read_csv(os.path.join(report_direct, 'ds5_findings_and_impressions_wo_ds_more_syn.csv'))

    data_with_labels = pd.DataFrame(columns=['id', 'image_id', 'text', 'label'])
    i = 0
    missing_reports = 0

    num_0 = 0
    num_1 = 0
    num_2 = 0
    num_3 = 0
    num_4 = 0

    balance_to = 155

    for file in all_files:

        file_check = file[0:13]

        # checks to see if there is an image file and a text file and puts the name of the image with the text and label
        #if reports_1['id'].str.contains(file_check).any():
        #    text = get_text(reports_1, file_check)
        #    if num_0 < balance_to:
        #        data_with_labels.loc[i] = [file_check, file, text, 0]
        #        num_0 += 1
        #    else:
        #        i = i - 1
        if reports_2['id'].str.contains(file_check).any():
            text = get_text(reports_2, file_check)
            if num_1 < balance_to:
                data_with_labels.loc[i] = [file_check, file, text, 0]
                num_1 += 1
            else:
                i = i - 1
        elif reports_3['id'].str.contains(file_check).any():
            text = get_text(reports_3, file_check)
            if num_2 < balance_to:
                data_with_labels.loc[i] = [file_check, file, text, 1]
                num_2 += 1
            else:
                i = i - 1
        elif reports_4['id'].str.contains(file_check).any():
            text = get_text(reports_4, file_check)
            if num_3 < balance_to:
                data_with_labels.loc[i] = [file_check, file, text, 2]
                num_3 += 1
            else:
                i = i - 1
        #elif reports_5['id'].str.contains(file_check).any():
        #    text = get_text(reports_5, file_check)
        #    if num_4 < balance_to:
        #        data_with_labels.loc[i] = [file_check, file, text, 4]
        #        num_4 += 1
        #    else:
        #        i = i - 1
        else:
            missing_reports += 1
            i = i - 1

        i += 1

    data_with_labels.set_index("id", inplace=True)
    return data_with_labels


# gets the text from the reports which matches the file_check argument
def get_text(reports, file_check):
    row = reports.loc[reports['id'] == file_check]
    text = row['text']
    text = text.iloc[0]
    return text

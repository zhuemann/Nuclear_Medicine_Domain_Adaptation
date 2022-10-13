from os import listdir
from os.path import isfile, join
from os.path import exists
import numpy as np
import pandas as pd
import os

def get_id_label_dataframe(dir_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/"):

    #negative_dir = 'Z:/Lymphoma_UW_Retrospective/Data/mips/Group_1_2_3_curated'
    #negative_dir = '/home/zmh001/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Data/mips/Group_1_2_3_curated'
    #positive_dir = 'Z:/Lymphoma_UW_Retrospective/Data/mips/Group_4_5_curated'
    #positive_dir = '/home/zmh001/r-fcb-isilon/research/Bradshaw/Lymphoma_UW_Retrospective/Data/mips/Group_4_5_curated'

    negative_dir = os.path.join(dir_base, 'Lymphoma_UW_Retrospective/Data/mips/Group_1_2_3_curated')
    positive_dir = os.path.join(dir_base, 'Lymphoma_UW_Retrospective/Data/mips/Group_4_5_curated')

    # gets all the file names in and puts them in a list
    neg_files = [f for f in listdir(negative_dir) if isfile(join(negative_dir, f))]
    pos_files = [f for f in listdir(positive_dir) if isfile(join(positive_dir, f))]

    if "Thumbs.db" in neg_files: neg_files.remove("Thumbs.db")
    if "Thumbs.db" in pos_files: pos_files.remove("Thumbs.db")

    print(f"num 0 labels: " + str(len(neg_files)))
    print(f"num 1 labels: " + str(len(pos_files)))
    print(f"fraction 0/(0+1): " + str(len(neg_files) / (len(neg_files) + len(pos_files))))

    # creates all the labels for each file
    labels = []
    for i in range(0, len(neg_files)):
        labels.append(0)
        # labels.append([1,0])

    for i in range(0, len(pos_files)):
        neg_files.append(pos_files[i])
        labels.append(1)
        # labels.append([0,1])

    df = pd.DataFrame()

    df['image_id'] = neg_files
    df['labels'] = labels

    # df = five_class_setup()

    return df

def get_text_id_labels(dir_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/"):

    report_files = ['ds123_findings_and_impressions_wo_ds_more_syn.csv', 'ds45_findings_and_impressions_wo_ds_more_syn.csv' ]

    #report_direct = '/home/zmh001/r-fcb-isilon/research/Bradshaw/Zach_Analysis/text_data/'
    report_direct = os.path.join(dir_base, 'Zach_Analysis/text_data')
    df = pd.DataFrame(columns=['id', 'text'])
    for i, file in enumerate(report_files):
        df0 = pd.read_csv(os.path.join(report_direct, file))
        if len(report_files) > 2:  # multi-class, one-hot
            label_i = [0] * len(report_files)
            label_i[i] = 1
            df0['label'] = [label_i] * len(df0)
        else:  # binary
            df0['label'] = i

        df = pd.concat([df, df0], axis=0, join='outer')

    # get list of image files
    df_images = get_id_label_dataframe(dir_base = dir_base)
    df_images["image_id"] = df_images["image_id"].str.replace(r'_mip.png', '')

    df = df.merge(df_images, left_on='id', right_on='image_id')
    df = df.drop_duplicates(subset=["id"])
    # print(df)
    # print(df.head())
    # df.head()

    return df

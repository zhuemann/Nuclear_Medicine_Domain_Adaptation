# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from vit_train import vit_train
from classification_model_bert_unchanged import fine_tune_model, test_saved_model
from get_id_label_dataframe import get_id_label_dataframe
from multimodal_classifcation import multimodal_classification
from make_u_map import make_u_map
from u_map_embedded_layers import multimodal_u_maps
from five_class_setup import five_class_image_text_label
import pandas as pd
import os

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    #vit_train()

    local = False
    if local == True:
        directory_base = "Z:/"
    else:
        directory_base = "/home/zmh001/r-fcb-isilon/research/Bradshaw/"

    DGX = True
    if DGX == True:
        directory_base = "/UserData/"

    # test = five_class_image_text_label()
    # print(test)


    # multimodal_classification(dir_base = directory_base, n_classes = 3)
    # multimodal_u_maps(dir_base = directory_base)
    # make_u_map()

    Fine_Tune = False
    if Fine_Tune == True:
        fine_tune_model(
            model_selection=2,  # 0=bio_clinical_bert, 1=bio_bert, 2=bert
            num_train_epochs=3,
            test_fract=0.2,
            valid_fract=0.1,
            truncate_left=True,  # truncate the left side of the report/tokens, not the right
            n_nodes=768,  # number of nodes in last classification layer
            vocab_file='',  # leave blank if no vocab added, otherwise the filename, eg, 'vocab25.csv'
            report_files=['ds123_findings_and_impressions_wo_ds_more_syn.csv',
                          'ds45_findings_and_impressions_wo_ds_more_syn.csv']
        )


    #test_mat = [[1,0,0], [0,1,0], [0,0,1]]
    #seeds = [456,915,1367, 712]
    #seeds = [712]
    #seeds = [1555, 1779, 2001, 2431, 2897, 3194, 4987, 5693 ,6386]
    seeds = [117,295,98,456,915,1367,712]
    accuracy_list = []
    for seed in seeds:
        #filepath = 'Z:/Zach_Analysis/result_logs/confusion_matrix_seed' + str(seed) + '.xlsx'
        #print(directory_base)
        #filepath = os.path.join(directory_base, '/UserData/Zach_Analysis/result_logs/confusion_matrix_seed' + str(seed) + '.xlsx')
        #print(filepath)
        #df = pd.DataFrame(test_mat)
        #df.to_excel(filepath, index=False)

        acc, matrix = multimodal_classification(seed=seed, batch_size=3, epoch=40, dir_base=directory_base, n_classes=5)
        accuracy_list.append(acc)

        df = pd.DataFrame(matrix)
        ## save to xlsx file
        filepath = os.path.join(directory_base, '/UserData/Zach_Analysis/result_logs/for_abstract/bio_clinical_bert/confusion_matrix_seed' + str(seed) + '.xlsx')

        df.to_excel(filepath, index=False)

    print(accuracy_list)


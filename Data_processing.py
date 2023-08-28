import os
import shutil
import pandas as pd


def json_labels():
    with open("groundtruth.json") as file:
        jfile = pd.read_json(file)
    return jfile


def copy_train_img(train_path, lbl, df):

    for img in df[lbl]['ok']:
        image_path = os.path.join('./images', img)
        shutil.copy(image_path, train_path)
    for img in df[lbl]['good']:
        image_path = os.path.join('./images', img)
        shutil.copy(image_path, train_path)


def copy_test_img(test_path, lbl, df):

    for img in df[lbl]['query']:
        image_path = os.path.join('./images', img)
        shutil.copy(image_path, test_path)


def create_folder(folder, lbl_list, df, flag):
    for lbl in lbl_list:
        path = os.path.join(folder, lbl)
        if not os.path.isdir(path):
             os.makedirs(path)
        if flag:
            copy_train_img(path, lbl, df)
        else:
            copy_test_img(path, lbl, df)


jlabel = json_labels()
jlabel = jlabel.drop(['junk'])

label_list = jlabel.columns.tolist()
create_folder('train', label_list, jlabel, True)
create_folder('test', label_list, jlabel, False)

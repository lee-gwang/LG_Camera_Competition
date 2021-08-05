# - preprocess.py - #

from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import os
import tqdm
import cv2

def load_data():
    train = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')
    return train, test

def cut_img(img_path_list, save_path, stride, img_size):
    os.makedirs(f'{save_path}{img_size}', exist_ok=True)
    list_=[]
    #num = 0
    for path in tqdm.tqdm(img_path_list):
        img = cv2.imread(path)
        img_id = path.split('/')[-1].split('.')[0].split('_')[-1]
        num = 0
        for top in range(0, img.shape[0], stride):
            for left in range(0, img.shape[1], stride):
                piece = np.zeros([img_size, img_size, 3], np.uint8)
                temp = img[top:top+img_size, left:left+img_size, :]
                piece[:temp.shape[0], :temp.shape[1], :] = temp
                np.save(f'{save_path}{img_size}/{img_id}_{num}.npy', piece)
                list_.append(f'{save_path}{img_size}/{img_id}_{num}.npy')
                num+=1
                
    return list_
          

def preprocess_train(img_size=512):
    train = pd.read_csv('./data/train.csv')

    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    train['fold'] = -1
    for n_fold, (_,v_idx) in enumerate(kf.split(train)):
        train.loc[v_idx, 'fold'] = n_fold

    train['input_img']  = './data/train_input_img/'+train['input_img'] 
    train['label_img'] = './data/train_label_img/'+train['label_img']
    img_id_ = train['img_id']


    train_input_files = train[train['fold']!=3]['input_img'].tolist()
    train_label_files = train[train['fold']!=3]['label_img'].tolist()

    val_input_files = train[train['fold']==3]['input_img'].tolist()
    val_label_files = train[train['fold']==3]['label_img'].tolist()


    #
    list_1 = cut_img(train_input_files, './data/train_input_img_', img_size//2, img_size)
    list_2 = cut_img(train_label_files, './data/train_label_img_', img_size//2, img_size)
    list_3 = cut_img(val_input_files, './data/val_input_img_', img_size//2, img_size)
    list_4 = cut_img(val_label_files, './data/val_label_img_', img_size//2, img_size)


    #
    temp_train = pd.DataFrame()
    temp_test = pd.DataFrame()
    #
    temp_train['input_img'] = list_1
    temp_train['label_img'] = list_2
    temp_train['type_'] = 'train'
    #
    temp_test['input_img'] = list_3
    temp_test['label_img'] = list_4
    temp_test['type_'] = 'val'

    df = pd.concat([temp_train, temp_test], 0).reset_index(drop=True)
    df['img_id'] = df['input_img'].apply(lambda x : x.split('/')[-1].split('_')[0])
    df.to_csv(f'./data/preprocess_train_{img_size}.csv', index=False)
def preprocess_test(img_size=512):
    test = pd.read_csv('./data/test.csv')

    test['input_img']  = './data/test_input_img/'+test['input_img'] 
    test_input_files = test['input_img'].tolist()
    #
    list_1 = cut_img(test_input_files, './data/test_input_img_', img_size//2, img_size)
    #
    temp_test = pd.DataFrame()
    #
    temp_test['input_img'] = list_1
    temp_test['img_id'] = temp_test['input_img'].apply(lambda x : x.split('/')[-1].split('_')[0])
    temp_test.to_csv(f'./data/preprocess_test_{img_size}.csv', index=False)


if __name__ == '__main__':
    preprocess_train(img_size=512)
    preprocess_train(img_size=768)
    #preprocess_test(img_size=768)


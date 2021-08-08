import segmentation_models_pytorch as smp
from models.model import SRModelsIf
import torch.cuda.amp as amp
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import tqdm
import argparse
import zipfile
import os
import cv2
import time
# ------------------------
#  Arguments
# ------------------------
parser = argparse.ArgumentParser(description='Inference')
parser.add_argument('--gpu', type=str, default= '0', help='gpu')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
def predict_new(stride=640, batch_size=64, img_size=768):
    """inference function

    Args:
        stride : stride size
        batch_size : inference batch size
        img_size : inference image size
    """
    # test data load
    test = pd.read_csv('./data/test.csv')
    img_paths = f'./data/test_input_img/'+test['input_img']

    # model load
    results = []
    model = SRModelsIf().to(device)
    model = nn.DataParallel(model)
    model.eval()

    # inference start
    with torch.no_grad():
            with amp.autocast():
                for img_path in img_paths:
                    img = cv2.imread(img_path)
                    img = img.astype(np.float32)/255
                    crop = []
                    position = []
                    batch_count = 0

                    result_img = np.zeros_like(img)
                    voting_mask = np.zeros_like(img)

                    for top in tqdm.tqdm(range(0, img.shape[0], stride)):
                        for left in range(0, img.shape[1], stride):
                            piece = np.zeros([img_size, img_size, 3], np.float32)
                            temp = img[top:top+img_size, left:left+img_size, :]
                            piece[:temp.shape[0], :temp.shape[1], :] = temp
                            crop.append(piece)
                            position.append([top, left])
                            batch_count += 1
                            if batch_count == batch_size:
                                crop = torch.from_numpy(np.array(crop)).permute(0,3,1,2).to(device)

                                pred = model(crop)*255
                                pred = pred.detach().cpu().numpy()
                                crop = []
                                batch_count = 0
                                for num, (t, l) in enumerate(position):
                                    piece = pred[num]
                                    h, w, c = result_img[t:t+img_size, l:l+img_size, :].shape
                                    result_img[t:t+img_size, l:l+img_size, :] += piece[:h, :w]
                                    voting_mask[t:t+img_size, l:l+img_size, :] += 1
                                position = []
                    if batch_count != 0: 
                        crop = torch.from_numpy(np.array(crop)).permute(0,3,1,2).to(device)

                        pred = model(crop)*255
                        pred = pred.detach().cpu().numpy()
                        crop = []
                        batch_count = 0
                        for num, (t, l) in enumerate(position):
                            piece = pred[num]
                            h, w, c = result_img[t:t+img_size, l:l+img_size, :].shape
                            result_img[t:t+h, l:l+w, :] += piece[:h, :w]
                            voting_mask[t:t+h, l:l+w, :] += 1
                        position = []
                        
                    result_img = result_img/voting_mask
                    result_img = np.around(result_img).astype(np.uint8)
                    results.append(result_img)
        
    return results
def make_submission(result):
    sub_imgs = []
    sub_dir = './submission/'
    os.makedirs(sub_dir, exist_ok=True)
    os.chdir(sub_dir)
    for i, img in enumerate(result):
        path = f'./test_{20000+i}.png'
        cv2.imwrite(path, img)
        sub_imgs.append(path)
    submission = zipfile.ZipFile(f"./submission.zip", 'w')
    for path in sub_imgs:
        submission.write(path)
    submission.close()

if __name__ == '__main__':
    start = time.time()
    test_result = predict_new(stride=640, batch_size=64, img_size=768)
    make_submission(test_result)
    print(f'{time.time()-start:2f}초 걸림')

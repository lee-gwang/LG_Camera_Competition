# LG_Camera_Competition
```
​```
${LG_Folder}
├── train.py
├── inference.py
├── preprocess.py
├── utils.py
├── dataloader.py
|
├── saved_models
|   └── 46e_31.9065_s.pth
|   └── 41e_32.1909_s.pth
|
├── submission
|   └── submission.zip
|
├── data
|   └── train_input_img
|   └── train_label_img
|   └── test_input_img
| 
├── requirements.txt
└── submission_final.csv 
​```
```

## Preprocess Script
```bash
$ python preprocess.py
```

## Training Script
```bash
$ python train.py --gpu=0,1 --img_size=512 --batch_size=64 --exp_name=512_models
$ python train.py --gpu=0,1 --img_size=768 --batch_size=32 --exp_name=768_models
```

## Inference Script
```bash
$ python inference.py --gpu=0,1
```

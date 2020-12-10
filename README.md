# CS_T0828_HW3

## Reference from Github
 detectron2: https://github.com/facebookresearch/detectron2
## Environment & Hardware
 - OS win10
 - Python=3.6.5
 - torch=1.6.0 and torchvision=0.7.0
 - CUDA=10.1
 - CuDNN=7.6
 - GCC=9.2
 
 - CPU: AMD Ryzen R5-5600X
 - RAM: 16G*2
 - GPU: NVIDIA GTX1060 6G
 
 ## Files Discription
 The uploaded files are without Detectron2. If you wnat to run, you need to build dtectron2 first.
  - **demo.py**: After you build detectron2, you can run this file to check whether you build it successfully. This is modified from the original detectron2 provided demo.py.
  - **hw2_main.py**: This is main file for HW3, it contains training and testing part. Some hyperparameters can be modified here, like epoch, batch size, learning rate, etc.
  
  NOTE: different from hw2, because annotation of this assignment is already COCO format, we don't need to convert it.
  
  ## Reproducing Submission
    +-Detectron2
    |  +-dataset
    |  | +-train_annotaion
    |  | +-test_annotation
    |  | +-train_images
    |  | | +-images
    |  | +-test_images
    |  | | +-images
    |  +-output
    |  | +-my model
    |  +-ImageNet pretrained model
    |  +-hw3_main.py
    |  +-demo.py
    |  +-submission file
   To reproduce my submission, you can follow following steps:
   #### 1. Install detectron2
   As mentioned above, files I uploaded didn't contain detectron2. So you need to install it first, you can find the link at the top. Please follow the steps in link . Also create folder and put my file into detectron2's folder just like above. 
   #### 2. Download Pretrained model
   You need to download the pretrained model to retrain it. After you run the code, if you don't have the model, the program will download it automatically. But I suggested you download from [Detectron2 Model Zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md). Here are also many other model you can choose.
   ##### warnings: because the dataset is Tiny PASCAL VOC, so I suggested you don't use the model is pretrained from COCO or VOC.
   #### 3. Preparing dataset
   You can download thte dataset from [here](https://drive.google.com/drive/folders/1fGg03EdBAxjFumGHHNhMrz2sMLLH04FK). And please create the folder like I build above.
   #### 4. Run
   Now, you can run my code hw3_main.py. I have already divided them into several block.
   In cell 1, here is import necessray libraries.
   In cell 2, it's the training part.
   In cell 3, it's the testing part.
  
  ## Brief Introduction
   The major task of this assignment is to do instance segmentation on the provided Tiny PASCAL VOC dataset. There are only 1349 training images and 100 test images which contains 20 common object classes. I use Detectron2 to train Mask R-CNN with ResNext101+FPN backbone. Finally, I got the 0.58576 mask mAP (with IoU = 0.5). And there are some successful examples.
  <img src="https://github.com/kyliao426/CS_T0828_HW3/blob/main/example%20images/2008_003141.jpg" width="50%" height="50%"><img src="https://github.com/kyliao426/CS_T0828_HW3/blob/main/example%20images/2010_001908.jpg" width="50%"  height="50%"> 
  
  ## Methodology
  ![image](https://i.imgur.com/R5mKkNj.png)
  </br>
    I used Detectron2 on this task. Detectron2 is a very strong and useful tool.  I can use the code structure which is very similar to the last assignment with only a few modifications. 
    At first, I used the Mask R-CNN with ResNet50+FPN backbone. And I got the 0.52989 on mAP. But the score of this assignment is only base on mAP, so I decided to try to train another model with the backbone “ResNext101”, which is more complex than “ResNet50”. But because of the limit of VRAM, I reduced the batch size from 2 to 1, and doubled the iterations to try to achieve the same epoch. And other hyperparameters are the same. Finally, I got 0.58576 on mAP. I periodically recorded my model and tested it, and the result is shown below. As you can see in Table.1, The best accuracy occurs at the 170000 iterations. There are also some hyperparameters configuration in Table.2

  | iterations | testing accuracy (IoU=0.5) |
  | ---- | ---- |
  | 40000 | 0.53006 |
  | 60000 | 0.50902 |
  | 80000 | 0.56500 |
  | 100000 | 0.54378 |
  | 110000 | 0.53554 |
  | 120000 | 0.55789 |
  | 130000 | 0.56200 |
  | 140000 | 0.54141 |
  | 150000 | 0.54560 |
  | 160000 | 0.56990 |
  | 170000 | 0.58576 |
  | 180000 | 0.53517 |
  | 190000 | 0.54366 |
  | 200000 | 0.50398 |
  
   Table.1


  | hyperparameters | value |
  | ---- | ---- |
  | batch size | 1 |
  | iterations(total) | 200000 |
  | iterations(best) | 170000 |
  | learnung rate | 0.00025 |
 
  Table.2
  
  ## Findings and Summary
   As mentioned above, I first used the Mask R-CNN with ResNet50+FPN backbone. At first, I expect I can improve my result through this method easily. However, I got falling instead of rising score, 0.50398 in mAP. The first thing I think is there is the overfitting, so I use the model which is stored during half the duration. And I got the 0.54378 on mAP, then I test all the models I saved during training and got the Table1. After observing and thinking about it, I think this is overfitting. As you can see, the testing accuracy reached the highest at the iteration 170000, and then dropped down. I think in such a small dataset like this, if we want to solve overfitting, finding the appropriate epoch is more important than doing data augmentation. Another thing is that in addition to the significant increase in training time, though I didn’t measure the exact time, I obviously felt that the prediction time has also become very long. If we really want to apply it in reality, we need to have a choice between time and accuracy.

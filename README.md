
## Data prepration

### COCOA dataset proposed in [Semantic Amodal Segmentation](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhu_Semantic_Amodal_Segmentation_CVPR_2017_paper.pdf).

1. Download COCO2014 train and val images from [here](http://cocodataset.org/#download) and unzip.

2. Download COCOA annotations from [here](https://github.com/Wakeupbuddy/amodalAPI) and untar.

3. Ensure the COCOA folder looks like:

    ```
    COCOA/
      |-- train2014/
      |-- val2014/
      |-- annotations/
        |-- COCO_amodal_train2014.json
        |-- COCO_amodal_val2014.json
        |-- COCO_amodal_test2014.json
        |-- ...
    ```

4. Create symbolic link:
    ```
    cd deocclusion
    mkdir data
    cd data
    ln -s /path/to/COCOA
    ```

### KINS dataset proposed in [Amodal Instance Segmentation with KINS Dataset](http://openaccess.thecvf.com/content_CVPR_2019/papers/Qi_Amodal_Instance_Segmentation_With_KINS_Dataset_CVPR_2019_paper.pdf).

1. Download left color images of object data in KITTI dataset from [here](http://www.cvlibs.net/download.php?file=data_object_image_2.zip) and unzip.

2. Download KINS annotations from [here](https://drive.google.com/drive/folders/1hxk3ncIIoii7hWjV1zPPfC0NMYGfWatr?usp=sharing) corresponding to [this commit](https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset/tree/fb7be3fcedc96d4a6e20d4bb954010ec1b4f3194).

3. Ensure the KINS folder looks like:

    ```
    KINS/
      |-- training/image_2/
      |-- testing/image_2/
      |-- instances_train.json
      |-- instances_val.json
    ```

4. Create symbolic link:
    ```
    cd deocclusion/data
    ln -s /path/to/KINS
    ```

## 
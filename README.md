# Object Detection in an Urban Environment

## Project overview

This project shows that we can use transferlearning in tensorflow to train a convolutional neural network on a new dataset to detect objects.
Object detection is very important for innovative technologies like autonomous cars. To fully understand a dynamic environment it is super important to detect and classify all objects in the visual input. Without this capability we cannot naviagte in complex environment like the traffic in an urban environment. We want to train a neural network to detect cars/vehicles, pedestrians and cyclists.
We will especially highlight the need of data augmentation for the training of such a neural network. 

## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/).

[OPTIONAL] - The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. We have already provided the data required to finish this project in the workspace, so you don't need to download it separately.

## Set up

### Project Structure

The project was done in a virtual workspace with a GPU. The required python libraries can be found in the ``` requirements.txt ``` file.

#### Data

The data you will use for training, validation and testing is organized as follow:
```
/home/workspace/data/
    - train: contain the train data (86 tfrecord files)
    - val: contain the val data (10 tfrecord files)
    - test - contains 3 files to test your model and create inference videos
```

#### Experiments
The experiments folder will be organized as follow:
```
/home/workspace/experiments/
    - pretrained_model/
    - exporter_main_v2.py - to create an inference model
    - model_main_tf2.py - to launch training
    - reference/ - reference training with the unchanged config file
    - experiment0/ - create a new folder for each experiment you run
    - experiment1/ - create a new folder for each experiment you run
    - experiment2/ - create a new folder for each experiment you run
    - label_map.pbtxt
    ...
```

## Prerequisites

### Local Setup [OPTIONAL] 

You can directly pull the initial git project from here: https://github.com/udacity/nd013-c1-vision-starter and follow the instructions there.

For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites.

### Download and process the data [OPTIONAL] 

**Note:** ”If you are using the classroom workspace, we have already completed the steps in the section for you. You can find the downloaded and processed files within the `/home/workspace/data/preprocessed_data/` directory. Check this out then proceed to the **Exploratory Data Analysis** part.

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file.

You can run the script using the following command:
```
python download_process.py --data_dir {processed_file_location} --size {number of files you want to download}
```

You are downloading 100 files (unless you changed the `size` parameter) so be patient! Once the script is done, you can look inside your `data_dir` folder to see if the files have been downloaded and processed correctly.

### Classroom Workspace

In the classroom workspace, every library and package should already be installed in your environment. You will NOT need to make use of `gcloud` to download the images.

## Instructions

### Exploratory Data Analysis

You should use the data already present in `/home/workspace/data/waymo` directory to explore the dataset! This is the most important task of any machine learning project. To do so, open the `Exploratory Data Analysis` notebook. In this notebook, your first task will be to implement a `display_instances` function to display images and annotations using `matplotlib`. This should be very similar to the function you created during the course. Once you are done, feel free to spend more time exploring the data and report your findings. Report anything relevant about the dataset in the writeup.

Keep in mind that you should refer to this analysis to create the different spits (training, testing and validation).


### Create the training - validation splits
In the class, we talked about cross-validation and the importance of creating meaningful training and validation splits. For this project, you will have to create your own training and validation sets using the files located in `/home/workspace/data/waymo`. The `split` function in the `create_splits.py` file does the following:
* create three subfolders: `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`
* split the tf records files between these three folders by symbolically linking the files from `/home/workspace/data/waymo/` to `/home/workspace/data/train/`, `/home/workspace/data/val/`, and `/home/workspace/data/test/`

Use the following command to run the script once your function is implemented:
```
python create_splits.py --data-dir /home/workspace/data
```

### Edit the config file

Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf).

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `/home/workspace/experiments/pretrained_model/`.

We need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 2 --checkpoint /home/workspace/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map /home/workspace/experiments/label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

### Training

You will now launch your very first experiment with the Tensorflow object detection API. Move the `pipeline_new.config` to the `/home/workspace/experiments/reference` folder. Now launch the training process:
* a training process:

```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```

* for the model with augmentation in the pipeline:

```
python experiments/model_main_tf2.py --model_dir=experiments/experiment1/ --pipeline_config_path=experiments/experiment1/pipeline_new.config
```
Once the training is finished, launch the evaluation process. Launching evaluation process in parallel with training process will lead to OOM error in the workspace:
* an evaluation process (for the pipeline with augmentation):
```
python experiments/model_main_tf2.py --model_dir=experiments/experiment1/ --pipeline_config_path=experiments/experiment1/pipeline_new.config --checkpoint_dir=experiments/experiment1/
```

**Note**: Both processes will display some Tensorflow warnings, which can be ignored. You may have to kill the evaluation script manually using
`CTRL+C`.

To monitor the training, you can launch a tensorboard instance by running
`npython -m tensorboard.main --logdir experiments/reference/`. 

### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup.

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it.

**Important:** If you are working on the workspace, your storage is limited. You may to delete the checkpoints files after each experiment. You should however keep the `tf.events` files located in the `train` and `eval` folder of your experiments. You can also keep the `saved_model` folder to create your videos.


### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:

```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/experiment1/pipeline_new.config --trained_checkpoint_dir experiments/experiment1/ --output_directory experiments/experiment1/exported/
```

This should create a new folder `experiments/reference/exported/saved_model`. You can read more about the Tensorflow SavedModel format [here](https://www.tensorflow.org/guide/saved_model).

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/experiment1/exported/saved_model --tf_record_path data/test/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/experiment1/pipeline_new.config --output_path experiments/experiment1/animation3.gif
```

## Dataset
### Dataset analysis
Each tfrecord files contains frames, captured by a camera. We can iterate over the data and query a lot of information. Each frames contans the following labels: 

- 'image'
- 'source_id'
- 'key'
- 'filename'
- 'groundtruth_image_confidences'
- 'groundtruth_verified_neg_classes'
- 'groundtruth_not_exhaustive_classes'
- 'groundtruth_boxes'
- 'groundtruth_area'
- 'groundtruth_is_crowd'
- 'groundtruth_difficult'
- 'groundtruth_group_of'
- 'groundtruth_weights'
- 'groundtruth_classes'
- 'groundtruth_image_classes'
- 'original_image_spatial_shape'

We can visualize the frame as shown here and with the additional information we can plot the groundtruth boxes, which label our objects in the image.

![](/images/imagesDataSet.JPG)

To get an initial overview on the data I checked how often each class appeared in 30000 sampled frames. We can see that vehicles appeared 519214 times, pedestrians 146336 times and cyclists 3730 times. This is an issue for the training, because the data set is highly imbalanced. This will lead to a better learning of the vehicle class compared to the other classes. 

![Count of each class in 30000 frames.](/images/ClassCount.JPG)

The distribution of the count of frames per class count underlines this issue. While the pedestrians and cyclists have the highest number of frames without an appearance, the vehicle distribution shows a different picture. Here we have only a small amount of frames without any vehicles. It is also notable that the range of values differs. While we have at most 5 cyclists in a frame, we have more than 60 vehicles and more than 40 pedestrians in some frames.

![](/images/ClassDist.JPG)

### Cross validation

A total of 99 tfrecord files is used. A split of 0.869, 0.101 and 0.03 is used for the training, testing and validation sets. We use the validation and testing set to check if our model overfits.

Because of the limitations in the workspace, it was not possible to run the training and evaluation process in parallel. Instead we evaluated the performance of the models after the training. The evaluation loss should be close to the training loss, if we want to prevent overfitting.

## Training
### Reference experiment

The reference model only has a horizontal flip and random crop image option for the data augmentation. What stand out is that the loss accumulated quite high in the befinning before dropping gradually, even tho we are using a pretrained neural network. By the shape of the loss functions it looks like we are improving the model as the loss get minimized.

![](/images/TB_reference_loss.JPG)

Checking on the evaluation metrics after 2500 epochs reveals that this model is really bad. All metrics have a value close to zero. Only the Detection Box Recall on large images has a value of around 0.04. Which is still super bad.

![](/images/TB_reference_precision.JPG)

![](/images/TB_reference_recall.JPG)

To sum up the results, we cannot detect anything with this model! The evaluation metrics supports this finding. So we need to change things up.

Things that can quickly be improved for this model:

- increase the batch size
- add more data augmentation to the pipeline, especially to train for night vision/different weather conditions

### Improve on the reference
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.

For the new pipeline I increased the batch size to 8. And I added the following data augmentations:
- random_rgb_to_gray
- random_adjust_brightness 
- random_adjust_contrast 
- random_adjust_saturation 
- random_distort_color
- random_black_patches 

The RGB to gray augmentation makes the model robust for day and night vision. We also want to adjust the brightness, as we have a mix of very bright and dark weather conditions. The contrast needs also to be adjusted, as we have different levels across the data. We also adjust the saturation randomly and distort the color to make the model more independent from the color of the objects/environment. To make the model more robust for camera input noise, we add random black patches.

The specifications are taken from [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto).

An example of the augmented input can be seen here:

![](/images/dataAugmentation.JPG)

I increased the total training steps to 4000, but the training was interrupted due the limited time on the workspace. So I only trained this model for around 2275 steps before it terminated. Still we could improve the reference model a lot.

What stands out is that we have a higher initial loss for the new model without a large increase. Instead it declines sharply.

![](/images/TBLossesExp1.JPG)

If we compare the loss to the reference model we can see that the best loss of the reference model is around 0.7, while our new model has a loss of less than 0.15!

![](/images/TBLosses.JPG)

We could improve the mean average precision (mAP) to 0.11, the mAP for large boxes to 0.42, to 0.41 for medium sized boxes and to 0.04 for small sized boxes. The Intersection over Union (IoU) for 0.5 threshold is 0.21, while the IoU for 0.75 threshold is 0.10.

This reveals that the model struggle mostly with small objects and that we still could improve the IoU of our detection capability. The precision for large and medium sized objects with around 0.42 is quite good for such a small training period, but we still need to improve the precision for small objects.

![](/images/TBDetectionBoxesPrecision.JPG)

The average recall (AR) given 1 detection per image is 0.027. The AR given 10 detections per image is 0.11, the AR given 100 detections per image is 0.16 and for AR@100 (large) is 0.54. The AR@100 (medium) is 0.52 and for AR@100 (small) is 0.10.

This results indicate that our model still has a lot of problems to detect all objects/classes in the image. The greater the objects in the image or detections per image are, the better our model.

![](/images/TBDetectionBoxesRecall.JPG)

We can see that the learning rate is identical, we only decrease it over 4000 instead of 2500 steps for the new model.

The better performance of our model comes to the expense of the computation time. We only manage to calculate 0.45 steps per second on average, while the reference model could perform around 1.3 steps per second.

![](/images/TBLrSteps.JPG)


### Visualization of the result model

We can create an animation of the trained model as explained in the setup. We can see that the model needs some time before it can detect moving objects, especially at night. We also see that is does not detect every object, especially if it is partly behind some other object.

![](/images/animation1.gif)

![](/images/animation2.gif)

![](/images/animation3.gif)

### Final remarks

I am quite happy with the result of my object detection model. Given more resources and time, we could improve our initial results for sure. Things to investigate:

- rebalance the sampling for the training data to tackle the imbalanced dataset
- finetune the hyperparameters
- use a different learning rate scheduler
- investigate more data augmentation methods with different parameters
- increase the batch size of the pipeline. As indicated in the article: [What's the Optimal Batch Size to Train a Neural Network?](https://wandb.ai/ayush-thakur/dl-question-bank/reports/What-s-the-Optimal-Batch-Size-to-Train-a-Neural-Network---VmlldzoyMDkyNDU) gives some insights on the reasoning behind this claim.
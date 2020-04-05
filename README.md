# DeepFake Detection CSE 240 Project


## Major directories
1. Augmented_image_file_label: extracted and preprocessed images are stored here (with respect to file label dataset).
2. Augmented_image_machine_label: extracted and preprocessed images are stored here (with respect to pre-trained Xception generated label dataset).
3. Classification: codes are here!
4. Dataset: the raw training dataset named as train_sample_videos, our used test dataset named as test_new_videos.
5. Submission: output prediction will be stored here.

## How to run
1. Install required packages.
2. Direct to "classification" directory.
3. Need to download the training dataset to dataset directory (https://www.kaggle.com/c/deepfake-detection-challenge/data).
4. Run data_preprocessing, resize_data_augmentation, file starts with "train", prediction in order.

## Major files
1. data_preprocessing: use pre-trained Xception model to generate labels for extracted images from each training video. Store the corresponing machine learning and file label as well.
2. detect_from_video: from the below references (pytorch).
3. resize_data_augmentation: resize the image into 64*64 shape and do data augmentation.
4. statistics: calculate statistics related problems.
4. train_: various kinds of training models with respect to 4 kinds of training dataset.
5. pre_trained_model_prediction: use a pre-trained model to make the prediction directly (pytorch).
6. prediction: load trained models from directory "trained_models" to make prediction on test dataset.


## References
Refered part of codes from FaceForensics github [here](https://github.com/ondyari/FaceForensics/tree/original). 


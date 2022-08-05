# Realtime Mask Detection

Machine Learning Project purely meant for educational purpose

## Problem statement

In today's pandemic world, it is crucial to monitor the people in the world. It is a necessity to wear mask to prevent the spread of contagious communicable diseases that spread through the air, especially CoVID-19. A need to maintain personal hygiene to avoid spread of disease has motivated people to wear masks. But, not all people follow the norm. So, the application at hand monitors the people at realtime to detect if a person wears a mask or not.

## Install Dependencies

If you have already installed `anaconda`, you can create a virtual environment using the following commands:

```bash
conda create --name env
conda activate env
```

Make sure to install the python packages in requirements.txt:

```bash
pip install -r requirements.txt
```

## Dataset

- Sourced from:
  - Kaggle
  - Google Images
  - Other open source image libraries
- Credits: Balaji Srinivasan

## Repository Structure

- `dataset/` contains the dataset for training the model
  - `with_mask\` contains images of people wearing mask
  - `without_mask\` contains images of people NOT wearing mask
- `face_detector/` contains the model that detects the face, which the region of interest to detect the mask
  - `deploy.prototxt` is the protobuffer text file that contains details regarding the face detection model
  - `res10_300x300_ssd_iter_140000.caffemodel` is the model used to detect faces
- `Train.py`: contains the code to preprocess the images in the dataset and train the model to detect masks
- `Test.py`: contains the code to start a video stream to perform a realtime mask detection on the live stream

## References

1. [Face Mask Detection](https://www.youtube.com/watch?v=Ax6P93r32KU) by Balaji Srinivasan - **YouTube** Tutorial and Walkthrough of code
2. [Face-Mask-Detection](https://github.com/balajisrinivas/Face-Mask-Detection) by Balaji Srinivasan - **Github** Repository
3. [Tensorflow installation (with CUDA, cudNN and GPU)](https://www.youtube.com/watch?v=hHWkvEcDBO0) by Aladdin Perrson - **YouTube** Tutorial

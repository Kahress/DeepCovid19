# DeepCovid19
Siamese Network architecture applied to Pneumonia and Covid-19 detection

Project developed for the course [DD2424 Deep Learning in Data Science](https://www.kth.se/student/kurser/kurs/DD2424?l=en) at [KTH](https://www.kth.se)

## Problem

This project was developed in Spring 2020, at the beginning of the pandemic situation due to Covid-19 in Europe. Quick techniques to recognize infected patients were investigated, but the available data about this infection was limited. Nevertheless, chest X-Ray images of patients with common pneumonia were available. 

## Description 

In this project, we develop a Siamese architecture with Deep Convolutional Networks for the detection of pneumonia from Chest X-Ray Images from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

The intuition behind the chosen architecture is learning similar latent representations of X-Ray Images showing pneumonia and maximizing their distance with the latent representations of healthy chests.

Transfer learning can then be applied to detect Covid-19 (as a form of Pneumonia) just by fine-tuning the network with a few available x-ray images of patients with Covid-19. 

The model was implemented both with PyTorch and Tensorflow Keras and trained making use of the computational resources of Google Cloud

## Examples of the data

![Example of a Chest X-Ray Image with Pneumonia](https://storage.googleapis.com/kagglesdsdata/datasets/17810/23812/chest_xray/train/PNEUMONIA/person1004_virus_1686.jpeg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20220129%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220129T132158Z&X-Goog-Expires=345599&X-Goog-SignedHeaders=host&X-Goog-Signature=0b2ebabc2c5c2d94941857804f7b0c35de40f6a4b2d8415523d058017f8e5546b3d937e931af681ec335785542db6cdbc62a94fe9ab291d86495243c8e018aaff1fe5db8120a995d932fde8bc7ca8a9b44eec36a2ab9a674d8bb75fcba7e67459df7d7fade311d785952d7c9c720a89ed5e4ab2396f9b20e64a6dfd3e1e55a6054720be200b104996a88e643cd2aaf5385e91253d927f1c7a53f8eb2517745c2e61ce7a2b97170ebf919eb6e7da5da4f2fa9b7ec0b00fcbd1a319c12d10efca704c925ee4a1d22b6461359ae86cf48ff0b7c9f9f47c3da9bae302ee0c1083624fb29c1199450c7a89d699c409e47256180142484b9f5f9824a5240dc82561ad8)

Due to the heterogeneous properties of the images (size, intensities,...), previous image preprocessing was required.

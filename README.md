# Dogs Vs. Cats: Training a Covnet Using a Pretrained Model

Convolutional neural networks (CNN) are now one of the easiest and best machine learning methods to classify images. In this project, I have trained a CNN to classify whether an image is a picture of a cat or a dog.

The model is built using TensorFlow Keras, and mainly uses `Conv2D` layers and `MaxPooling2D` layers.

This notebook expands on the code in Chollet's *Deep Learning with Python* 2E.

## Building a CNN Model

Training a CNN from scratch can yield decent results, but it turns out that we can utilize much larger pretrained models. In this project, we are using the VGG16 model as a base, and then building on top of that.

## Training the CNN Model

To train this model, I am using the full dogs vs cats data set from Kaggle. This data set contains 25,000 images of cats and dogs. Training this model takes about 1.5 hours even when utilizing Google Colab's GPU runtimes. 

## Fine Tuning a Pretrained Model

We use the model from our original training to further fine tune our model for the best possible results. 

Fine-tuning consists of unfreezing a few of the top layers of a frozen model base used for feature extraction, and jointly training both the newly added part of the model (in this case, the fully connected classifier) and these top layers. This is called fine-tuning because it slightly adjusts the more abstract representations of the model being reused, in order to make them more relevant for the problem at hand.

## Results

By training on a large dataset, and leveraging a pretrained model, we are able to achieve 97.5% accuracy, which I must say is pretty impressive.
# Landmark Classification Project: CNN and Transfer Learning

Welcome to the Landmark Classification project, where we delve into the world of image recognition. We'll be accomplishing this through three distinct phases, each documented in its dedicated Jupyter Notebook:

## 1. Developing a Landmark Classifier from Scratch using CNN

In this preliminary phase, we embark on a journey to create a Convolutional Neural Network (CNN) from scratch for the purpose of landmark classification. Here's what's in store:

1. **Exploring the Dataset**: We begin by comprehensively examining the dataset, gaining insights into its characteristics, and understanding the distribution of classes.

2. **Data Preprocessing**: We prepare the dataset for model training by resizing images, normalizing pixel values, and partitioning the data into training and validation sets.

3. **Constructing the CNN Architecture**: The core of this step involves architecting a CNN model. This entails integrating convolutional layers, pooling layers, and fully connected layers.

4. **Model Training and Evaluation**: We proceed to train the constructed CNN on the training set while evaluating its performance on the validation set. Metrics such as accuracy and loss are monitored.

5. **Exporting with Torch Script**: The trained model is exported using Torch Script, paving the way for potential deployment.

## 2. Leveraging Transfer Learning for Landmark Classification

Moving forward, we explore the realm of transfer learning and its potential for our landmark classification task. Here's how it unfolds:

1. **Surveying Pre-trained Models**: We undertake an extensive survey of pre-trained models designed by experts for similar image classification tasks.

2. **Selecting an Apt Pre-trained Model**: Among the pre-trained models, we carefully select a suitable candidate that aligns with our dataset and objectives.

3. **Fine-tuning and Training**: The chosen pre-trained model is fine-tuned for our specific task by adjusting its final layers and training it on our dataset.

4. **Assessing Model Performance**: We evaluate the performance of the transfer-learned model on the validation set, drawing comparisons with the CNN developed from scratch.

5. **Exporting with Torch Script**: Similar to the previous phase, we export the optimized transfer learning model using Torch Script for prospective deployment.

## 3. Implementing a Landmark Classifier App

In this final phase, we translate our model into a user-friendly app that empowers users to predict landmarks depicted in their images. The sequence unfolds as follows:

1. **App Development**: We engage in the development of an intuitive app that facilitates users in uploading images for landmark predictions.

2. **Integration of the Model**: Our trained model, be it the scratch-built CNN or the transfer-learned model, is integrated into the app to provide real-time predictions.

3. **Testing and Reflecting**: Extensive testing of the app is conducted using various images. Concurrently, we reflect upon the model's strengths and limitations in practical scenarios.

This comprehensive project is captured in three distinct Jupyter Notebooks, meticulously detailing the journey from constructing a model from scratch to harnessing transfer learning and culminating in an interactive app. Each phase is a testament to the power of neural networks in enhancing image recognition capabilities. Feel free to delve into the notebooks to explore the intricacies of the process. Happy learning and building! 🌆📸

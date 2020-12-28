## Traffic Sign Recognition

Overview
---
This project trains a convolutional neural networks to classify traffic signs. Training and validation occurs on the traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, I will try out the model on images of German traffic signs that I found on the web.

The code for training and testing the CNN is included in the `Traffic_Sign_Classifier.ipynb`.


The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This project requires the following Python libraries installed:

- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [TensorFlow](http://tensorflow.org)
- [Matplotlib](http://matplotlib.org/)
- [Pandas](http://pandas.pydata.org/)

### Dataset

1. [Download the dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip) and unzip it to `.\data\`. This is a pickled dataset in which the sign images are resized to 32x32 and are organized in training, validation, and test sets.

2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/sbonab/CarND-Traffic-Sign-Classifier
cd CarND-Traffic-Sign-Classifier
jupyter notebook Traffic_Sign_Classifier.ipynb
```

3. Follow the instructions in the `Traffic_Signs_Recognition.ipynb` notebook.


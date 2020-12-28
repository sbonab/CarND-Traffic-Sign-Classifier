# **Traffic Sign Recognition** 

## Writeup

---

**Building a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Report/signs.png 
[image2]: ./Report/distribution.png
[image3]: ./Report/grayscale.png
[image4]: ./Report/architecture.png
[image5]: ./Report/web_signs.png
[image6]: ./Report/image_01.png
[image7]: ./Report/image_02.png
[image8]: ./Report/image_03.png
[image9]: ./Report/image_04.png
[image10]: ./Report/image_05.png
---

This is the project writup and the link to the github repository is [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

#### 1. Basic summary of the dataset.

I used the numpy library to calculate the summary statistics of the traffic
signs data set:

* The size of the training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The image shows 50 randomly chosen images from the training set.

![alt text][image1]

The following bar chart shows the distribution of the classes for the training, test, and validation set, respectively.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Preprocessing the image data

As a first step, I decided to convert the images to grayscale. This is mainly because in the [reference article](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), it is mentioned that suprisingly converting images to grayscale results in better classification. I do this by getting the inner product of the image channel numbers with [0.2989,0.5870,0.1140]. This is equivalent to converting the images to YUV space and only choosing the Y. 

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

As a last step, I normalized the image data so that the data will have mean close to zero and equal variance. I have used a simple normalization method by using (pixel - 128)/128 equation.

I decided not to augment the training set by generating additional data since the architecture shows a good classification performance just by using the training set as is.

#### 2. Final model architecture: model type, layers, layer sizes, connectivity, etc.

The model architecture I implemented is similar to the following image, outlined in the [reference article](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).

![alt text][image4]

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Average pooling	    | 2x2 stride, same padding, outputs 14x14x6 	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Average pooling	    | 2x2 stride, same padding, outputs 5x5x16  	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 1x1x400 	|
| RELU					|												|
| Concatenation		    | Output of the second and the third layers     |
| Dropout     		    | With the probability of keep_prob = 0.5       |
| Fully connected		| Inputs: 800, outputs: 200                     |
| RELU					|												|
| Dropout     		    | With the probability of keep_prob = 0.5       |
| Fully connected		| Inputs: 200, outputs: 43                      |
 


#### 3.Training the model

To train the model, I used the `softmax_cross_entropy_with_logits()` function to calculate the loss function based on the generated logits. I also use the Adam optimizer to optimize the weight and bias variables. To help prevent overfitting, I have used the dropout method with the probability of `keep_prob = 0.5` for keeping some of the model variables. The hyperparameter values are 

| Hyperparameter | Value |
|:--------------:|:-----:|
|Epochs          | 25    |
|Batch size      | 256   |
|Learning rate   | 0.001 |

#### 4. More discussion on the taken approach

My final model results were:
* training set accuracy of 99.8 %
* validation set accuracy of 95.7 % 
* test set accuracy of 94.3 %

I started with using the lenet-5 architecture initially by changing the final connected layer  output from 10 to 43. However, I couldn't get close to higher accuracies using this architecture since it is well-tuned fo digit classification of only 10 classes. 

Then, I decided to implement the architecture discuss in the [reference article](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). This architecture also uses multi-scale features where the output of the first stage is branched out and added to the output of the second stage and then is fed to the fully-connected layers. This also helps increase the accuracy of the model classification. 

In the next step, I have modified the hyperparameters to increase the accuracy while keeping the training time within a reasonable limit. 
 

### Test a Model on New Images

#### 1. Some random examples from the Internet

Here are five German traffic signs that I found on the web. Note that I have resized the images to `(32x32x3)`.

![alt text][image5]

Obviously, I have chosen those signs that are included in the 43 different types that are included in the training set. The images fairly cover the whole signs are taken from a good angle so it is expected that the classifier should work well on these example images.

#### 2. Discussion the model's predictions 

Here are the results of the model:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children Crossing 	| Children Crossing              				| 
| No entry     			| No entry 										|
| Speed limit (70 km/h) | Speed limit (70 km/h)             			|
| Yield          		| Yield      					 				|
| Bumpy Road			| Bumpy Road      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.3%. 

Moreover, for each one, I have plotted the top 5 model prediction. Note that the y axis is in logscale. 


#### 3. Top 5 Softmax Probabilities

The code for making predictions on my final model is located in the  Ipython notebook.

The top 5 softmax probabilities for either of the images are shown below

![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/sign1_big.png "Original"
[image3]: ./examples/sign1_processed.png "Processed"
[image4]: ./examples/sign1_big.png "Traffic Sign 1"
[image5]: ./examples/sign2_big.png "Traffic Sign 2"
[image6]: ./examples/sign3_big.png "Traffic Sign 3"
[image7]: ./examples/sign4_big.png "Traffic Sign 4"
[image8]: ./examples/sign5_big.png "Traffic Sign 5"

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used Python to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is 32 x 32.
* The number of unique classes/labels in the data set is 43.

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a plot of random signs from the training data set.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

As a first step, I decided to convert the images to grayscale because this will fasten the computation since we need three times less data and also will force the CNN to concentrate on shapes rather than on colors.

As a second step, I normalized the image data because the optimizer has a much easier work to do if the input data is well conditioned.

Here is an example of a traffic sign image before and after preprocessing:

![alt text][image2]![alt text][image3] 

I decided to not generate additional data because the CNN achieved sufficient accuracy.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

|      Layer      |               Description                |
| :-------------: | :--------------------------------------: |
|      Input      |         32x32x1 grayscale image          |
|   Convolution   | 1x1 stride, valid padding, outputs 28x28x6 |
|      ReLU       |                                          |
|   Max pooling   | 2x2 stride,  valid padding, outputs 14x14x6 |
|   Convolution   | 1x1 stride, valid padding, outputs 10x10x16 |
|      ReLU       |                                          |
|   Max pooling   | 2x2 stride, valid padding, outputs 5x5x16 |
|     Flatten     |               outputs 400                |
| Fully connected |               outputs 120                |
|      ReLU       |               with dropout               |
| Fully connected |                outputs 84                |
|      ReLU       |               with dropout               |
| Fully connected |                outputs 43                |



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I kept number of epochs (10) and batch size (128) as suggested in the course material. Increasing one or both of them hasn't given a much better accuracy, so I decided to stay with intial values to keep things simple and fast. To speed up a little bit I chose to increase the learning rate to 0.0015.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.991.
* validation set accuracy of 0.958.
* test set accuracy of 0.934.

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  I started with LeNet which was introduced in course chapter "LeNet In TensorFlow".
* What were some problems with the initial architecture?
  The architecture had an output size of 10, because it was used for MNIST data. Also it didn't achieve sufficient accuracy in the beginning.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  First I changed output size to 43. Secondly after trying to tune several other parameters, I decided to introduce dropout layers at different places. It turned out that adding two of them after the last two ReLUs achieved the best results and boosted the accuracy beyond 0.93.
* Which parameters were tuned? How were they adjusted and why?
  I tried to tune several parameters like epochs, batch size, image preprocessing, color instead of grayscale and so on. But the model remained under 0.93 accuracy, until I added two dropout layers. Dropout rate was chosen to 0.5 and it worked out of the box.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  A convolution layer makes the prediction independant from sign position. A dropout layer make the prediction robust against distortions.

If a well known architecture was chosen:
* What architecture was chosen?
  LeNet
* Why did you believe it would be relevant to the traffic sign application?
  LeNet was designed for image recognition (hand-written numbers), which can also be used to detect a set of traffic signs.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] ![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because it contains some shadows on the white area.
The second image might be difficult to classify because it has low contrast.
The third, fourth and fifth images might be difficult to classify because there are several similar signs.
The fifth image might be difficult to classify because its bottom edge is not aligned horizontally.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

|            Image            |         Prediction          |
| :-------------------------: | :-------------------------: |
|            Yield            |            Yield            |
|        Priority road        |        Priority road        |
| Dangerous curve to the left | Dangerous curve to the left |
|         Bumpy road          |         Bumpy road          |
|          Road work          |          Road work          |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

Almost all images are predicted with a certainty of almost 100%. Only the third image is difficult for the model. The model is relatively sure that this is a "Dangerous curve to the left" sign (probability of 0.986), and the image does contain a "Dangerous curve to the left" sign. The top five soft max probabilities are:

| Probability |         Prediction          |
| :---------: | :-------------------------: |
|    .986     | Dangerous curve to the left |
|    .009     |        Slippery road        |
|    .005     |      Bicycles crossing      |
|    .000     |  Road narrows on the right  |
|    .000     |     Beware of ice/snow      |

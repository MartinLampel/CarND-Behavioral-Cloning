# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/Architecture.png "Model Architecture"
[image2]: ./img/loss.jpg "Loss"
[image3]: ./img/centers_mask.jpg "Image before cropping"
[image4]: ./img/center_cutout.jpg "Image after cropping"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---

## Project structure

* model.py Code to train a neural network for the Behavioral Cloning task
* drive.py The code to drive the car in autonomous mode
* model.h5 The weights of the trained neural network
* video.py Convert recorded images into a video
* run.mp4 The recorded car in autonomous mode. 

### Usage 

To train the neural network run 
```
python model.py
```

Change the argument of the `load_samples` function to the path of the driving_log.csv.

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

To record the autonomously car execute

```
python drive.py model.h5 run
```

The images then are stored in the `run` folder. 

## Data Collection

The first track is driven forwards and in reverse dircection to achieve a better generalization of the first track. Additionally there are image sequences where the drive off the center to left or right and the correction to back to center was recorded. Another image sequence was that we start off the left/right  and drive to the center of the lane. 

To improve further the generalization there are several rounds on the second track recorded. 


## Model Architecture and Training Strategy

### 1. Model Architecture 

The model architecture is based on the [Blog End-to-End Deep Learning for Self-Driving Cars](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/)

  
<p align="center">
  <img src="./img/Architecture.png ">
</p>



Follow adaptions are made to the architecture:

* Normalization layer 

  A layer which normalizes each color channel 
* Cropping 

  The recorded images show then entire scence, which includes the sky, the landscape, the road. To train the neural network do drive around the track, such informations 
  as the landscape or the sky are not relevant.
  
  Therefore I decided to cut out an area of the image with the important information for the training. This is done by cutting 70 pixels from the top and 25 pixels from   the bottom. The following images show an image of the center before and after the cropping. The red area shows the area which remains after the cropping was done.

  ![alt text][image3]  |  ![alt text][image4]
  
After the normalization and cropping, five convolutional layers are following. The filter kernel for each convolutional layer is a (5,5) kernel with a (2,2) stride. 
Then 4 fully connected layers after the convolutional layers are used to give the new steering angle. 
  

### 2. Training Process

#### Data Augmentation

In the original approach by NVIDEA the applied a rotation and shift to the images. 

I used the center, left, and right images without a rotation or shift. To use the left and right images, we change the steering angle of these images with a correction factor.
I tested various values in the range between -0.5 and 0.5 and achieved with values of 0.25 for the left camera and -0.25 for the right camera good results. 

This increases the training samples by a factor of 3. During the training randomly a left, right, or center image is selected. The steering angle adds the corresponding correction factor. 

The next step was to apply a flip of the selected image. This data augmentation doubles the size of the training data. The steering angle must also be flipped by multiplication with a -1.

The generation of the training samples is done with the generator function `generator`. In this function implements the previous described steps. 

#### Model Parameters and Overfitting

To prevent overfitting I used the simple approach to limit the number of epochs. This limit is set to 5 epochs. 
 ![alt text][image2] 

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

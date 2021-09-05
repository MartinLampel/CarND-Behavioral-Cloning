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
  
The function `create_model` creates the model for the described architecture. 


### 2. Training Process

#### Data Augmentation

In the original approach by NVIDEA the applied a rotation and shift to the images. 

I used the center, left, and right images without a rotation or shift. To use the left and right images, we change the steering angle of these images with a correction factor.
I tested various values in the range between -0.5 and 0.5 and achieved with values of 0.25 for the left camera and -0.25 for the right camera good results. 

This increases the training samples by a factor of 3. During the training randomly a left, right, or center image is selected. The steering angle adds the corresponding correction factor. 

The next step was to apply a flip of the selected image. This data augmentation doubles the size of the training data. The steering angle must also be flipped by multiplication with a -1.


When working with datasets that have a large memory footprint python generators are a convenient way to load the dataset one batch at a time rather than loading it all at once. The generation of the training samples is done with the generator function `generator`. In this function implements the previous described steps. 


#### Model Parameters and Overfitting

ADAM is used as optimizer for the training. The learning rate is not tuned. 

There are various methods to deal with overfitting such as early stopping, weights regularization, dropout. 

To prevent overfitting I used the simple approach to limit the number of epochs. This limit is set to 5 epochs.
Since we achieved with this method good results I implemented none of the other methods. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.
The size of validation set is 20 % of the data. 

As loss function for the optimizer the mean square error is selected. The results of the loss function over the 5 epochs can be seen in the next section.


## Results and Conclusion

I plotted the loss function over the epochs. The loss for the training decreases in each epoch. For the validation set, the loss decreases in each epoch except the 4th epoch. 

 ![alt text][image2] 
 
 
The video run.mp4 shows the car driving autonomously on the first track. It drives around the entire track without a critical situation. However, there are some situations, especially in curves, where the driving behavior can still be improved. 
Since the resulting behavior was sufficient the model is no further tuned.  

I've also tested the second track and it drives around but it works not good enough. Some situations are dangerous or where it leaves the road. The main reason for this is that there is not enough data for the second track available. The driving behavior can be improved that the car drives accident-free on the second track. Due to time constraints, I couldn't generate more data for the second track. This is a future task which I will do when I have enough time. 

To summarize our observations, the success for driving autonomously around the track depends more on the training data, than on the model architecture from my experience. 

I started by using simply the center images that results in worse performance on the track. As soon I used the left and right camera images, the behavior improved a lot. Finally, I saw another improvement by flipping the images. 

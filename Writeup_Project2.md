##* Traffic Sign Classifier - Udacity Selfdriving car Project 2

# Loading Data:

In this step we loaded the pickled data provided. The training, validation and test data are in the files train.p, valid.p and test.p. This data is presemt in the folder traffic-signs-data.

# Dataset Summary and Analysis:

In this step the data is analysed. The number of train, valid, test data and number of classes are counted and are printed.

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Training images data shape = (32, 32, 3)
Validation images data shape = (32, 32, 3)
Testing images data shape = (32, 32, 3)
Number of classes = 43


The data is then visualized using a bar chart. The number of images per class is plotted in the bar chart. This chart is saved as OriginalData_Visualization.jpg.

From the chart it is evident that the number of images for certain calsses is larger in number compared to the others. If the data is trained as it is, it will be bised towards these largers classes and will not be a good data set to train with. For this purposes, the data needs to be augmented.

For augmentation, classes with images less than 750 are considered. The images of this class are rotated by 10, -10, 15 and -15 degrees. These set of 4 rotated images are added 250 times. This increases the image quantity for classes with less than 750 images by 1000. After augmenting the data, it is then visualized using a bar chart. This chart is saved as AugmentedData_Visualization.jpg. The rotated images are saved to original training data set variables. To get access to these images easily (as augmenting the images might take some time) they are saved as X_train.npy and y_train.npy files. These files are loaded if notebook needs to restart and clear output. 
#* These files are not included in the solution because of their size 

## Preprocessing:

For preprocessing the data is shuffled and is given as input to the neural network.

## Model Architecture:

In this step a model is designed and trained to get validation accuracy of more than 93%.

The parameters which make the learning better are the number of EPOCHS, the batch size and the learning rate. 

The neural network has total 7 layers. The layers 1 through 4 are convolutional layers and are followed by Relu activation and max pooling. Layers 5 and 6 are fully connected layers and use Relu activation. Layer 7 is also fully connected and produces logits. 

# Layer1:Convolutional

Convolution:	Input: 32x32x3, Filter: 3x3, Output: 30x30x3
Activation:	Relu
Max Pooling:	Input: 30x30x3, Filter: 2x2, Output: 29x29x3

# Layer2:Convolutional

Convolution:	Input: 29x29x3, Filter: 4x4, Output: 27x27x6
Activation:	Relu
Max Pooling:	Input: 27x27x6, Filter: 2x2, Output: 26x26x6

# Layer3:Convolutional

Convolution:	Input: 26x26x6, Filter: 5x5, Output: 22x22x12
Activation:	Relu
Max Pooling:	Input: 22x22x12, Filter: 3x3, Output: 10x10x12

# Layer4:Convolutional

Convolution:	Input: 10x10x12, Filter: 5x5, Output: 6x6x18
Activation:	Relu
Max Pooling:	Input: 6x6x18, Filter: 2x2, Output: 3x3x18

# Layer5: Fully connected

The output of layer 4 is flattened and is given as input to this layer.

Input: 162, Output:100
Activation: Relu

# Layer6: Fully Connected

Input: 100, Output: 75
Activation: Relu

# Layer7: Fully Connected

Input: 75, Output:43 (logits)


## Training:

For training the parameters values selected are:

EPOCHS = 30
Batch Size = 75
Learning Rate = 0.0005
Optimizer: AdamOptimizer

After 30 epochs the validation accuracy is calculated to be 94.8%.

The test data accuracy is calculated to be 91.9%

The plot for training accuracy values and validation accuracy values are plotted. This plot is saved as Training_Plot.jpg.

## Training Approach:

Initially the data was tested with LeNet architecture without data augmentation. The validatation accuracy was calculated to be 89%. Next the data was augmented and trained using LeNet. The validation accuray was calculated to be 92.7%. 

To increase the validation accuracy to more than 93%, the architecture was changed. The seven layer architecture was desinged to be as close to LeNet as possible and give better results.

Initially the model was trained with 10 EPOCHS, 150 batch size and 0.001 learning rate. This yielded validation accuracy of 93.5%. To increase it more the parameters were changed to 30 EPOCHS, 50 batch size and 0.0005 learning rate. The results for these parameters is considered in this project. 

## Testing on Downloaded Images:

# Acquiring Images:

5 different traffic signs are downloaded from the internet to perform this test. The original images downloaded were 300x300 pixels and the traffic sign itself inside the picture was smaller. As the input for the designed architecture is 32x32 images, the downloaded images could not be given as input as they were. The second concern with the pictures was that, there was lot of other data in the picture including the traffic sign holder and the surroundings. If the image is resized and given as input to the neural net, there is lot of unwatend data that we train in the picture. In order to avoid this, the traffic sign is cropped from the picture and resized to 32x32 pixels. These resized and cropped images are given as input to the neural net to predict the labels. The details of the images downloaded is given below.

Image 1 - Stop
Image 2 - Speed limit(30 km/hr)
Image 3 - speed limit(60 km/hr)
Image 4 - Yield
Image 5 - Speed limit(50 km/hr)

These image are plotted. This plot is saved as Downloaded_Images.jpg. The original images are saved in the folder webImages.

# Performance:

The softmax predictions for the downloaded images are calculated and printed in the notebook.
The test accuracy for these images is found to be 80%. The trained model could identify 4 out 5 images. The training model wrongly identified image 3.

The top 5 softmax predictions are calculated and printed in the jupyter notebook.



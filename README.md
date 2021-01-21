# Multiclass-Image-classifier
Classifying retinal OCT images into one of 4 categories- NORMAL, CNV, DME or DRUSEN



Goals and motivation of analytics 
As we know, the life expectancy of humans is at an all-time high. People live longer than their ancestors due to increase in medical supplies, services, and technology. Hence, there is an increase in eye diseases as the degenerative effects related to eye health mainly increase with age. Along with that, humans are increasingly spending more time in front of a screen these days as a result of digitalization, which further adds to the problem of macular degeneration in eyes. There is a requirement to develop more efficient, quick and easy methods to detect eye diseases.
The motivation behind this project is to work on a technique which can be helpful to ophthalmologists to identify the retinal diseases more accurately and faster using image recognition technology. 
We can detect the most frequently occurring eye diseases, which should not take much time to detect as they are common. The eye diseases involved in this project are 


1)	AMD (Age related muscular degeneration): This disease is commonly seen in older people and it caused due to damaged retina or dead tissues. About 35% of adults who are above age of 75 around the world suffers from AMD.
 




2)	DR (Diabatic Retinopathy): Today diabetes is one of the most common disease seen in adults as well as youngsters in developing and developed countries because of improper intake of food and a lethargic lifestyle. That is the reason 80% of the people with diabetes around the world suffer from Diabatic Retinopathy. 
 


3)	Glaucoma: Glaucoma is caused by higher-than-normal pressure inside the eye because the eye is not able to get rid of some waste material a condition called ocular hypertension. In most types of glaucoma, optic nerve damage and vision loss occurs because the pressure inside the eye (IOP) is too high
 

We extracted the image dataset of the above-mentioned diseases and normal eye from Kaggle. Below is the link to access the dataset.
https://www.kaggle.com/paultimothymooney/kermany2018
Data Exploration
1)	Training:  83,484 images which were pre-classified as follows:
•	NORMAL – 26,315
•	CNV – 37,205
•	DME – 11,348
•	DRUSEN – 8,616
           Validation: 8 images in each category (32 total)
           Test: 242 images in each category (968 total) 
2)	We can see through the data that it is biased because the number of images in Normal and CNV category is way more than DME and DRUSEN.
3)	Images were not of uniform size or orientation.
4)	The ImageDataGenerator function from the keras library rescales and accounts for data which is rotated or horizontally flipped.
5)	Processed training data is fed to the CNN network in specified batch sizes and dimensions
6)	We used RandomUnderSampler function from imbalances-learn API to avoid the bias in the distribution of training images. The bar graph below shows the output of a random sample collected using the under sampler.
{0: 'Normal', 1: 'CNV', 2: 'DME', 3: 'DRUSEN'}


Transfer learning

•	When we have a pre-trained neural network that works well for an existing problem, we can utilize that model for a similar problem to save time and reduce the number of epochs needed to reach optimal accuracy for the model.
•	Generally, transfer learning is used when the problem we're transferring from has a huge amount of data and the problem we're transferring to has relatively less data. For example, a neural network trained to identify bone fractures or deformities can be used for the retina classification problem. 
•	If the data is small, we can get away with changing weights of only the last few layers but for big data, we generally change parameters for all the layers. This process of changing pre-trained weights is called fine-tuning in the context of deep learning

Models and algorithms
We used the concept of transfer learning and used models pre-trained on the ImageNet dataset. This dataset has more than 14 million images comprising of more than 20,000 categories.
1. VGG16 
•	VGG16 consists of convolutional layers, max pooling layers, activation layers, and finally fully connected layers. 
•	This network is characterized by its simplicity, using only 3×3 convolutional layers stacked on top of each other in increasing depth.
•	Reducing volume size is handled by max pooling. Two fully connected layers, each with 4,096 nodes are then followed by a softmax classifier


•	VGG16 gave us the following: The accuracy from this model is approximately 0.90 or 90%
•	Precision is just the ratio of correctly predicted positive observations (In this case CNN DME DRUSEN, Normal) to the total predicted positive observations.
•	This model does a pretty good job in terms of precision.
                                  

 

•	This model took approximately an hour to run for 9 epochs.
•	We used Adam optimizer, which is a adaptive learning rate optimization algorithm

2. Inception V3
•	Suppose, for example, a layer in our deep learning model has learned to focus on individual parts of a face. The next layer of the network would probably focus on the overall face in the image to identify the different objects present there. Now, to do this, the layer should have the appropriate filter sizes to detect different objects.
 
•	This is where the inception layer comes to the fore. It allows the internal layers to pick and choose which filter size will be relevant to learn the required information. So even if the size of the face in the image is different (as seen in the images below), the layer works accordingly to recognize the face. 
•	Inception Layer is a combination of 1×1 Convolutional layer, 3×3 Convolutional layer, 5×5 Convolutional layer, by concatenated their output into a single output vector. 
•	 The model expects images as input with the size 224 x 224 pixels with 3 channels.
 

•	InceptionV3 gave us the following: Confusion Matrix and Accuracy. The accuracy from this model is approximately 0.92 or 92%

•	The time taken by this model to run was approximately 45 minutes for 7 epochs.
•	The RMSprop optimizer was used for faster training.
•	3. ResNet50
 
•	ResNet50 has 50 layers, with each convolution block having 3 convolution layers and each identity block also having 3 convolution layers
•	In general, in a deep convolutional neural network, several layers are stacked and are trained to the task at hand. The network learns several low/mid/high level features at the end of its layers. 
•	In residual learning, instead of trying to learn some features, we try to learn some residual. Residual can be simply understood as subtraction of feature learned from input of that layer. 
•	ResNet50 does this using shortcut connections (directly connecting input of nth layer to some (n+x)th layer. It has proved that training this form of networks is easier than training simple deep convolutional neural networks and also the problem of degrading accuracy is resolved. This is the fundamental concept of ResNet50, and residual networks in general


Conclusion 
•	A Robust automatic image classifier method for OCT images that can detect the type of disease by observing the retinal layers.
•	Provides algorithm for detecting pathologies.
•	Classification model for the eye diseases. 
•	InceptionV3 algorithm is giving the best accuracy in least time.
•	Proposed automatic method shows best accuracy of 0.92 than other state-of-the-art methods.
Future Suggestions 
•	Can develop an automatic system for 2D and 3D segmentation method for OCT images
•	Software that can analyze fully automatically analysis of OCT images, extracting biomarkers and classifying the disease state. 
•	This software can be helpful in developing large scale clinical studies to find more significant biomarkers for understanding and monitoring the progression of the eye diseases.




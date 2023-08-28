# ImageRetrieval_Oxford5k

The task is to train a neural network that takes one image as input to output several of the most similar images from the database. The neural network was trained and tested on the Oxford5k dataset. <br/>

This set consists of 5062 images of 11 Oxford landmarks. For each image and landmark in the data set, one of four labels was generated: “good”, a good, clear image of the object; “ok” – more than 25% of the object is clearly visible; “bad” – there is no object; “junk” – less than 25% of the object is visible, or there is a very high level of overlap or distortion; "query" - requests. <br/>

All labels with their corresponding images are presented in the json file.

A Siamese neural network was used to solve the IR problem. <br/> 
Siamese networks consist of two identical neural networks, each with the same weights. Each network takes one of two input images as input. The outputs of the last layers of each network are then sent to a function that determines if the images contain the same identifiers. <br/>

Siamese network architecture
![Image alt](https://github.com/Marakuia/ImageRetrieval_Oxford5k/blob/main/inf/siamese)

The triplet loss was used as the loss function. The formula is shown below. <br/>
![Image alt](https://github.com/Marakuia/ImageRetrieval_Oxford5k/blob/main/inf/triplet)

Data processing is carried out in a Data_processing file. <br/>

Dependence of the error function on the epoch
![Image alt](https://github.com/Marakuia/ImageRetrieval_Oxford5k/blob/main/IR/loss_siamese)

Visualization of results
![Image alt](https://github.com/Marakuia/ImageRetrieval_Oxford5k/blob/main/IR/visual_siamese)

There was also a transfer of training to resnet-50 <br/>
Dependence of the error function on the epoch (pre-trained)
![Image alt](https://github.com/Marakuia/ImageRetrieval_Oxford5k/blob/main/IR_resnet/loss_resnet)

Visualization of results *pre-trained)
![Image alt](https://github.com/Marakuia/ImageRetrieval_Oxford5k/blob/main/IR_resnet/visual_resnet)

# RoboCrop: An automated classifier of soybean diseases

RoboCrop is a deep learning web application trained to identify eight disease categories for soybeans using images of diseased leaves. 

We sought to develop an application to democratize soybean disease identification and hopefully reduce the excessive use of pesticides. Over the course of two seasons of soybean production, we collected over 9,500 original field images of soybean leaves in eight distinct disease categories. We then used these original images to build and train a convolutional neural network-based automated classifier of digital images of soybean diseases.

Through our research approach we found that, for soybean foliar disease classification, the best performing base model was DenseNet201, using a from-scratch transfer learning approach. This was done using Tensorflow and Keras API. The details of this training model can be seen in the robocrop_cnn.py file. Using this model, we were able to develop an application that distinguishes between eight soybean disease/deficiency classes with an overall accuracy of 96.75%. 

The details of the research is currently being submitted for publication and we are looking at ways to make the 9500+ image dataset available for use.

## Usage

The model is hosted and available for use on our web application at [RoboCrop Webapp](http://sickbeans.skullisland.info/). Simply take a picture of a soybean leaf and upload it to the website. The model will return a soybean disease prediction.

The model is trained to recognize healthy plants, along with the following diseases: 

Bacterial Blight, Cercospora Leaf Blight, Downey Mildew, Frogeye, Potassium Deficiency, Soybean Rust, and Target Spot.

Example images of each disease category: 

![Soybean Diseases](Soy_Fig1.png)

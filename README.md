# clothing-recommender-webapp

![App demo](https://github.com/RomainBarraud/clothing-recommender-webapp/master/images/eight_screens.png)

## Authors
 - LEE, Sum Yin Sonny
 - BARRAUD, Romain
 - TANG, Yubo

## Project Description
This repository contains the source files of a recommender system for clothes in a web application.<br/>
A user will upload a photo and obtain in return a series of up to 3 clothes matching the style of the photo.<br/>

## Application workflow
1. The user access the webapp webpage
2. The user uploads a photo
3. A first model detects up to 18 body-points on the photo
4. The model returns 4 frames on the photo<br/>
&ensp; 4.1 Head<br/>
&ensp; 4.2 Upper body<br/>
&ensp; 4.3 Lower body<br/>
&ensp; 4.4 Whole body<br/>
5. A second model checks whether the head is a woman's or a man's
6. If 5. is true, the whole body, upper body and lower body images will sequentially go through a dedicated model analyzing the piece of cloth<br/>
&ensp; 6.1 Upper body clothing<br/>
&ensp; 6.2 Lower body clothing<br/>
&ensp; 6.3 Whole body clothing<br/>
7. Each model will return embedding vector representing the probabilities of belonging to a class of cloth
8. Each class will, crossed with the gender, will lead to display a relevant pool of available clothes. The chose photo in a given pool will be drawn randomly
9. The web app will display the uploaded photo, the same photo with the 3 frames 4.1, 4.2 and 4.3, and the 2 recommended pieces of clothes

## Installation
Reserve a server or cloud environment with an IP to open the app the world wide web. You can also run the application locally
Utilize One linux Ubuntu 18.04 instance
Run *pip install -r requirements.txt*
Download pose detection weights *chmod +x ./pose-detection/getModels.sh && ./getModels.sh*
Load the four h5 models, or rerun the notebooks, to classify respectively the faces, upper body clothes, lower body clothes and whole body clothes

## Infrastructure and environment
- Google Cloud
- Linux Ubuntu 18.04
- Python 3.7
- Flask-based

## To be done in future releases
- Add label tags in addition to classes to each piece of cloth
- Install each model on a different instance
- Use Nodejs as the central server coordinating all the other instances
- Install MongoDB to store uploaded photos and results

## Datasets
[Gender dataset](https://susanqq.github.io/UTKFace/)<br/>
[Deep Fashion dataset](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)

## References
[ISOM5240 – Deep Learning Applications for Business. HKUST by James S H KWOK](https://www.ust.hk)<br/>
[Gender age ethnicity classification](https://sanjayasubedi.com.np/deeplearning/multioutput-keras/)<br/>
[Posenet application](https://www.learnopencv.com/deep-learning-based-human-pose-estimation-using-opencv-cpp-python/)<br/>
[Women recommender system](https://medium.com/@sjacks/building-a-womens-fashion-recommender-2683856b97e3)<br/>
[Deep Fashion tuning](https://medium.com/@birdortyedi_23820/deep-learning-lab-episode-4-deep-fashion-2df9e15a63e1)<br/>
[Tensorflow](https://www.tensorflow.org/)<br/>
[Keras]( https://keras.io/)

## License
 - GNU GENERAL PUBLIC LICENSE

## Version
| Date       | Description   |
|:----------:|:--------------|
| 12/15/2019 |  Draft        |
| 12/19/2019 |  v1           |



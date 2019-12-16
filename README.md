# clothing-recommender-webapp

## Authors
- Romain BARRAUD
- Sonny
- Yubo TANG

## Project Description
This repository contains the source files of  recommender system for women clothes in a web application.

## Version
| Date       | Description   |
|:----------:|:--------------|
| 12/15/2019 |  Draft        |

## Typical workflow
1. The user access the webapp webpage
2. The user uploads a photo
3. A first model detects up to 18 body-points on the photo
4. The model returns 3 frames on the photo<br/>
&ensp; 4.1 Head<br/>
&ensp; 4.2 Upper body<br/>
&ensp; 4.3 Lower body
5. A second model checks whether the head is a woman's
6. If 5. is true, the upper body and lower body images will sequentially go through 2 other models<br/>
&ensp; 6.1 Cloth type<br/>
&ensp; 6.2 Cloth shape<br/>
&ensp; 6.3 Cloth textture
7. Each model 6.1, 6.2, 6.3 will return an embedding vector that will be concatenated into one embedding per photo
8. The embeddings will be sent a to a last models which will compute the distance with a database of clothes and return the closest for each photo.
9. The web app will display the uploaded photo, the same photo with the 3 frames 4.1, 4.2 and 4.3, and the 2 recommended pieces of clothes.

## Installation
Run *pip install -r requirements.txt*
Download pose detection weights *chmod +x ./pose-detection/getModels.sh && ./getModels.sh*


## Infrastructure and environment
- Linux Ubunut 18.04
- Google Cloud
- Full Python 3.7
- Flask-based

## To be done in future releases
- Install each model on a different instance
- Use Nodejs as the central server coordinating all the other instances
- Install MongoDB to store uploaded photos and results

## References
[Gender age ethnicity classification](https://sanjayasubedi.com.np/deeplearning/multioutput-keras/)<br/>
[Tensorflow data input generator](https://towardsdatascience.com/multi-label-image-classification-in-tensorflow-2-0-7d4cf8a4bc72)<br/>
[Women recommender system](https://medium.com/@sjacks/building-a-womens-fashion-recommender-2683856b97e3)<br/>

## License
 - GNU GENERAL PUBLIC LICENSE
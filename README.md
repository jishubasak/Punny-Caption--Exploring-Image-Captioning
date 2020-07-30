## Punny Captions - Automated Image Captioning
Author: Jishu Basak


## Problem Statement
Creating an Image Captioning application using Deep Learning algorithms


## Introduction:

Caption generation is a challenging artificial intelligence problem where a textual description must be generated for a given photograph.

It requires both methods from computer vision to understand the content of the image and a language model from the field of natural language processing to turn the understanding of the image into words in the right order.

Deep learning methods have demonstrated state-of-the-art results on caption generation problems. What is most impressive about these methods is a single end-to-end model can be defined to predict a caption, given a photo, instead of requiring sophisticated data preparation or a pipeline of specifically designed models.

## Conclusion and Future work

Note that due to the stochastic nature of the models, the captions generated may not be exactly similar to those generated in my case.

### Learnings:
Combination of deep learning models is a good approach towards solving complex Human-AI related problems. 

In this specific project I learned how the fusion of Convolutional Neural Networks with Recurrent Neural Network could actually emancipate text through images. 

Of course this is just a first-cut solution and a lot of modifications can be made to improve this solution like:

- Using a larger dataset.

- Changing the model architecture, e.g. include an attention module.

- Doing more hyper parameter tuning (learning rate, batch size, number of layers, number of units, dropout rate, batch normalization etc.).
- Use the cross validation set to understand overfitting. 

- Using Beam Search instead of Greedy Search during Inference.

- Using BLEU Score to evaluate and measure the performance of the model.


### References:

- https://cs.stanford.edu/people/karpathy/cvpr2015.pdf

- https://arxiv.org/abs/1411.4555

- https://arxiv.org/abs/1703.09137

- https://arxiv.org/abs/1708.02043

- https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/

- https://www.youtube.com/watch?v=yk6XDFm3J2c

- https://www.appliedaicourse.com/


You can explore the project in the Jupyter Notebook 

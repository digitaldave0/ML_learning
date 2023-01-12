# ML deep learning notes

## ML premier 

Artificial Intelligence (AI) is a broad term used to describe systems capable of making certain decisions on their own. Machine Learning (ML) is a specific subject within the broader AI arena, describing the ability for a machine to improve its ability by practising a task or being exposed to a large data set. Its fast moving set of technologies to see the latest look at https://paperswithcode.com/sota


[<img src="https://www.onemodel.co/hubfs/main-qimg-b6846724daf2af284a1137c1a8e72f56.png">]()


### Inputs

Think Food ingredients 
Recipe is the program 

### Outputs 

- Cooked food is the output (labels)
- With traditional programs you define the rules 
- For a complex problem you can't think about all the rules  that's when you use ML, ML is used when there are millions of rules,  like how to drive a = car, all the possibilities could never be mapped as rules
- Parameters are Patterns, numbers in data that make the model
- Deep learning models are problemistic , e.g there making a bet on that connection 
- Someone in YouTube wrote


> I think you can use ML for literally anything as long as you can turn it into numbers and then program it to find patterns, we write the algorithm and it finds the patterns  it can be literally be anything any input or output in the universe 


## Google's 1st rule of machine learning is

> If you can build a simple rules based system that doesn't require machine learning do that don’t use ML for rules based activities like a recipe 

# What deep learning is good for

- Use ML for problems that have a long list of rules. When the traditional approach fails machine learning, deep learning can help when there’s millions of rules like self driving cars

- Continually changing environments , deep learning can adapt and learn in new scenario's 

- Discovering insights with a large collection of data can you imagine trying to handcraft rules from the food 101 dataset for what 101 different foods looks like.

## What deep learning is not typically good for

- When you need explainability the patterns learned by a deep learning model are typically uninterpreted by a human weights and biases 
- When errors are unacceptable, since outputs from a deep learning model aren't always predictable 
- When the traditional approach is a better option, if you can accomplish what you want to achieve with simple rules based approach
- When you don't have much data, deep learning models usually require large amounts of data to produce great results
But we can see some techniques to will show how to get great results with small amounts of data given to the model
- Machine learning Vs deep learning 

## Structured data 

Since the advent of deep learning below have been termed as shallow algorithms 
- Use machine learning algorithms on structured data 
- ad gradient boosted machine,dlmc XGboost
- Random forest, tree based algorithm
- Gradient boost
- Naive Bayes 
- Nearest neighbour 
- Support vector machine 

## Unstructured data

- Is tweets, webpages, pictures, voice, we can turn unstructured data into a patterns, numbers using tensors
- Neural network 
- Fully connected neural network
- convolutional neural network
- Recurrent neural network
- Transformer

Depending on what your problem is, the algorithms listed above can be used to solve many problems, deterministics that have many millions of variables.

## What are neural networks

Neural networks are a type of machine learning algorithm inspired by the structure and function of the human brain. They are composed of interconnected 
"neurons" that can process and transmit information.


[<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/Neural_network_example.svg/1200px-Neural_network_example.svg.png" width="200">]()

Neural networks are trained to perform a specific task by adjusting the strengths of the connections between neurons, known as "weights." Once trained, a neural network can process inputs, make predictions, and learn from its mistakes, just like a human brain. Neural networks are commonly used for tasks such as image and speech recognition, language translation, and playing games. They are particularly useful for handling complex, non-linear relationships within data. 

### summary of neural networks 

Neural networks take in unstructured data, such as images or text, and convert them into a numerical representation, often called a tensor, which can be processed by the network. The network typically consists of an input layer, one or more hidden layers, and an output layer. The input layer receives the input data and passes it through the hidden layers, which apply transformations to the data using weights that have been learned through training. The output layer produces the final output, which can be a prediction, classification, or some other desired result.

### Hidden layers 

The number and size of the hidden layers, as well as the specific types of transformations applied to the data, depend on the problem being solved and the design of the neural network model. Different types of neural networks, such as convolutional neural networks and recurrent neural networks, are well-suited to different types of problems and data.

Overall, neural networks are a powerful tool for processing and analysing complex, unstructured data, and they have been widely used in many applications such as image and speech recognition, natural language processing, and predictive modelling

## The anatomy of a neural network

Here's a simple description of the anatomy of a neural network:

- Input layer: The input layer receives the input data and passes it on to the hidden layers for processing. The input layer has one or more neurons, depending on the complexity of the input data. Nodes

- Hidden layers: The hidden layers are sandwiched between the input and output layers and perform intermediate processing on the data as it flows through the network. Each hidden layer has one or more neurons, and the specific design of the hidden layers depends on the problem being solved and the specific neural network model being used. Neurons 

- Output layer: The output layer receives the processed data from the hidden layers and produces the final output of the network, which can be a prediction, classification, or some other desired result. The output layer has one or more neurons, depending on the complexity of the output. 

- Outputs 

Overall, a neural network is a network of interconnected "neurons" that process and transmit information, using learned weights to extract features and relationships from data and make predictions or perform other tasks. The specific design of the network, including the number and size of the hidden layers and the type of transformations applied to the data, depends on the problem being solved and the specific neural network model being used.

## Types of learning 

Different types of learning paradigms

### Supervised learning

In supervised learning, the algorithm is trained on a labelled dataset, where the correct output is provided for each example in the training set. The goal of the algorithm is to learn a function that maps inputs to their corresponding outputs. This function can then be used to make predictions on new, unseen examples. Examples of supervised learning tasks include classification, regression, and prediction

### Unsupervised learning

In unsupervised learning, the algorithm is not given any labelled examples and must discover the underlying structure of the data through exploration and analysis. The goal of the algorithm is to identify patterns and relationships in the data without the guidance of a labelled training set. Examples of unsupervised learning tasks include clustering, anomaly detection, and dimensionality reduction.

### Transfer learning

Transfer learning is a type of machine learning that involves using the knowledge and skills learned from solving one problem to improve the performance of a model on a different but related problem.

Transfer learning can be used to speed up the training process and improve the performance of a model by leveraging the knowledge learned from a pre-trained model that has already been trained on a large dataset. This can be particularly useful when the dataset for the new problem is small or when there are limited resources available for training a model from scratch.

There are two main types of transfer learning: feature-based transfer learning and fine-tuning. 

- In feature-based transfer learning, the features learned by the pre-trained model are extracted and used as input to a new model, which is trained on the new task using these features. 
- In fine-tuning, the pre-trained model is modified and re-trained on the new task, using both the pre-trained weights and new weights learned on the new task.
Transfer learning has been successfully applied to a wide range of machine learning tasks, such as image and text classification, natural language processing, and speech 

### Reinforcement Learning 

Reinforcement learning is a type of machine learning in which an agent learns by interacting with its environment and receiving feedback in the form of rewards or punishments. The goal of the agent is to maximise the cumulative reward it receives over time by taking actions that lead to desirable outcomes.

In reinforcement learning, the agent is faced with a series of states, and at each state, it can choose from a set of actions. The agent receives a reward or penalty based on its actions and the resulting state of the environment. 
The agent's learning process involves trial and error, as it tries different actions and receives feedback in the form of rewards or punishments. 

Through this process, the agent learns which actions are most likely to lead to positive outcomes and adjust its behaviour accordingly.

Reinforcement learning has been applied to a wide range of problems, including 

- control systems 
- game playing
- natural language processing. 

It is particularly useful for tasks where it is difficult to define a clear set of rules or a specific objective, and the agent must learn through trial and error how to interact with the environment in order to achieve a desired outcome.

## DNQ (Deep-Q-Learning)

Reinforcement Learning is an exciting field of Machine Learning that’s attracting a lot of attention and popularity. An important reason for this popularity is due to breakthroughs in Reinforcement Learning where computer algorithms such as Alpha Go and OpenAI Five have been able to achieve human level performance on games such as Go and Dota 2. One of the core concepts in Reinforcement Learning is the Deep Q-Learning algorithm. Naturally, a lot of us want to learn more about the algorithms behind these impressive accomplishments. In this tutorial, we’ll be sharing a minimal Deep Q-Network implementation (minDQN) meant as a practical guide to help new learners code their own Deep Q-Networks.

1. Reinforcement Learning Background
Reinforcement Learning can broadly be separated into two groups: model free and model based RL algorithms. Model free RL algorithms don’t learn a model of their environment’s transition function to make predictions of future states and rewards. Q-Learning, Deep Q-Networks, and Policy Gradient methods are model-free algorithms because they don’t create a model of the environment’s transition function.

2. The CartPole OpenAI Gym Environment

Figure 1: Balancing a pole in the CartPole Environment (Image by Author)
The CartPole environment is a simple environment where the objective is to move a cart left or right in order to balance an upright pole for as long as possible. The state space is described with 4 values representing Cart Position, Cart Velocity, Pole Angle, and Pole Velocity at the Tip. The action space is described with 2 values (0 or 1) allowing the car to either move left or right at each time step.

[<img src="https://miro.medium.com/max/720/0*Ch8Sq_3_b3Q1wVWa" width="250">]()

The CartPole environment is a simple environment where the objective is to move a cart left or right in order to balance an upright pole for as long as possible. The state space is described with 4 values representing Cart Position, Cart Velocity, Pole Angle, and Pole Velocity at the Tip. The action space is described with 2 values (0 or 1) allowing the car to either move left or right at each time step.



### What is deep learning actually used for?

Deep learning is a type of machine learning that involves using multi-layered artificial neural networks to learn and make decisions based on data. It has been successfully applied to a wide range of problems, including:

- Image and video processing tasks, such as object recognition and classification, image and video generation, and image and video style transfer.
- Speech and audio processing tasks, such as speech recognition, language translation, and music generation.
- Natural language processing tasks, such as language translation, text summarization, and sentiment analysis.
- Predictive modelling tasks, such as stock price prediction and customer churn prediction.
- Overall, deep learning has been used to achieve state-of-the-art results in many areas, and it has the potential to revolutionise the way we interact with and process data in a variety of fields

### Sequence-to-sequence (Seq2Seq)

Sequence-to-sequence (Seq2Seq) is a type of neural network architecture that is commonly used for tasks involving sequential data, such as natural language processing and machine translation.

In a Seq2Seq model, the input data is a sequence of items, and the goal is to predict a corresponding output sequence. For example, in the case of machine translation, the input sequence is a sentence in one language, and the output sequence is the translation of that sentence into another language.

Seq2Seq models typically consist of two main components: an encoder and a decoder. 

- The encoder processes the input sequence and produces a fixed-length representation, called the "context," which captures the key information in the input sequence. 
- The decoder then uses this context to generate the output sequence one item at a time, using the learned relationships between the input and output sequences.

Seq2Seq models have been successful in a variety of tasks involving sequential data, such as machine translation, language modelling, and chatbot development. They have the ability to handle variable-length input and output sequences, and they can learn complex relationships between the input and output data.

## Classification and regression

Classification and regression are two types of supervised learning tasks in machine learning.

> Classification is the task of predicting a categorical label, such as a class or a category, based on a given set of features. For example, a classification model might be trained to predict whether an email is spam or not spam based on the words and phrases it contains. Other examples of classification tasks include image classification, where the goal is to predict the class of an object in an image, and speech recognition, where the 
goal is to predict the words spoken in an audio recording.

> Regression is the task of predicting a continuous numerical value, such as a price or a probability, based on a given set of features. For example, a regression model might be trained to predict the price of a house based on its size, location, and other features. Other examples of regression tasks include weather forecasting, where the goal is to predict the temperature or precipitation at a given location and time, and stock price prediction, where the goal is to predict the future price of a stock based on historical data.

Overall, classification and regression are common types of supervised learning tasks that are widely used in a variety of applications. They involve training a model on a labelled dataset, where the correct output is provided for each example in the training set, and then using the trained model to make predictions on new, unseen examples.

## Transformers

Paper pubished 2017 when breakthrough happend

https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html#:~:text=Neural%20networks%2C%20in,with%20the%20Transformer.

Neural networks, in particular recurrent neural networks (RNNs), are now at the core of the leading approaches to language understanding tasks such as language modeling, machine translation and question answering. In “Attention Is All You Need”, we introduce the Transformer, a novel neural network architecture based on a self-attention mechanism that we believe to be particularly well suited for language understanding.

[<img src="https://miro.medium.com/max/1400/1*DIelATUu_zKd5eSqD8315Q.png" width="400">]()

https://www.youtube.com/watch?v=O3xbVmpdJwU

# PyTorch

### What's PyTorch

PyTorch is a popular open-source machine learning framework developed and maintained by Facebook's AI Research group. It is widely used for training deep learning models and has a number of features that make it well-suited for this purpose.
One of the main benefits of PyTorch is its intuitive and flexible programming model, which allows developers to easily define and manipulate complex neural networks. It also supports GPU acceleration, which allows for faster training of deep learning models.

In addition, PyTorch has a large and active community of users and developers, which provides a wealth of resources and support for those learning to use the framework. It is widely used in industry and academia, and has been used to achieve state-of-the-art results on a variety of machine learning tasks, including image and speech recognition, natural language processing, and predictive modelling.
Overall, PyTorch is a powerful tool for training deep learning models, and it is widely used in the field of machine learning due to its ease of use, flexibility, and strong performance.

## Install pyTorch

```bash Create an anaconda environment
conda create -n ml_learning python=3.7
conda activate ml_learning
# cpu only
conda install pytorch-cpu torchvision-cpu -c pytorch
# cuda core
conda install pytorch torchvision -c pytorch
Conda list
```
```code Testing pyTorch cpu only
import torch
# Set the device to "cpu"
device = "cpu"
# Check if GPU is available
print(torch.cuda.is_available())
# Create a random tensor on the device
a = torch.tensor([1, 2, 3], device=device)
# Perform some operations on the tensor
b = a + 2
c = b * 3
# Print the result
print(c)
````

Using Google Collab simplifies above

What is a tensor
What's a Tensor?

A tensor is a multi-dimensional array of data, often used in machine learning and deep learning to represent data and perform computations on it. 
Tensors are a fundamental data structure in PyTorch, and they are used to store and manipulate data in a variety of applications.
In PyTorch, tensors can be created and manipulated using the torch module. For example, you can create a tensor with a specific shape and data type using the following code:

```code
import torch
# Create a 3x3 tensor of floating-point values
tensor = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=torch.float32)
# Print the tensor
print(tensor)
```


Above will create a 3x3 tensor with floating-point values, which can be used for various machine learning and deep learning tasks.
A tensor processing unit (TPU) is a specialised hardware accelerator designed specifically for accelerating the training and inference of deep learning models. TPUs are developed and maintained by Google and are designed to work seamlessly with the TensorFlow framework. TPUs can significantly speed up the training process for deep learning models, particularly when used in conjunction with distributed training techniques.
Overall, tensors are an important data structure in PyTorch and other machine learning frameworks

The Maths in Machine Learning

https://youtu.be/CqOfi41LfDw

## Understanding the main Data Objects for Machine Learning

The tensor is a mathematical function from linear algebra that maps a selection of vectors to a numerical value. The concept originated in physics and was subsequently used in mathematics. Probably the most prominent example that uses the concept of tensors is general relativity.
In the field of machine learning, tensors are used as representations for many applications, such as images or videos. They form the basis for TensorFlow’s machine learning framework, which also takes its name from this.

### What are the Rank and Axis?

If you want to describe tensors more precisely, you need the so-called rank and the dimension. These can determine the size of the object. For this, we start with a matrix whose rank we want to find out. For this purpose, we form a simple Numpy array, which represents a matrix with three rows and three columns.

```code
import numpy as np 
matrix = np.array([
                   [1,5,7],
                   [2,9,4],
                   [4,4,3]
                  ])
```
The rank of a tensor now provides information about how many indices are needed to reference a single number in the element. It is also often referred to as the number of dimensions.
In the case of a matrix, this means that it has rank 2 (the matrix is two-dimensional) because we need two indices to output a specific number. Suppose we want to change from the object “Matrix” to the number 5 in the first row and second column, then we need two steps to get there.

```code
matrix[0]
array([1, 5, 7])
```

First, the first line with index 0 must be referenced:

```code
matrix[0]
array([1, 5, 7])
```

In the array that we get, we can select the second element with the index 1 (note: with Numpy, the count starts at 0!):

So matrices have rank 2 (are two-dimensional) because two indices are needed to get one number. However, there are many different sizes of matrices, for example with three rows and four columns or with five rows and two columns.
In order to distinguish different matrices and define them precisely, we use the so-called axes and their length. The axis of a tensor is the values in a specific dimension. For our example, “matrix” the Numpy array with the first index is an axis in the first dimension.


# First axis of the first dimension of the object "matrix"
```code
print(matrix[0])
[1 5 7]
```

The length of the axis in turn tells how many valid indices there are along the axis. In our case the length is three because on the first axis there are three indices in total (= the matrix has three rows). The following call leads to an error because the fourth index does not exist:

# Returns an error since there is no fourth index in "matrix"

```code
print(matrix[3])
```

IndexError: index 3 is out of bounds for axis 0 with size 3

Show error details


## What is the Size of a Tensor?
The size of a tensor gives the length of the axes for each dimension. This means that we can specify the size of our object “matrix” as follows:

```code
import tensorflow as tf
matrix = tf.convert_to_tensor(matrix)
tf.shape(matrix)
```

We get various information fed back from TensorFlow, which can be interpreted relatively easily. The tensor has the “Shape” 2, i.e. the rank 2 or two dimensions. We already expected this, since we know that a matrix is a 2-dimensional tensor. The “dtype” describes which data types are stored, in our case integers. 

Finally, we get the size of the tensor with “([3,3])”, because there are three different indices in each of the two dimensions. We had already expected this result since it is a matrix with three rows and three columns.

## What are the different Types of Tensors?

Tensors are the umbrella term for vectors and matrices and comprise multi-dimensional arrays in the machine learning field. Depending on the dimensionality of the array, a distinction is made between different types:

## Scalar

> The scalar, i.e. a single number, is a 0-dimensional tensor. No single index is needed to reference a number. The scalar, therefore, has no axes and therefore also the rank 0.
## Vector

> The vector is a 1-dimensional tensor and has rank 1. Depending on how many elements the vector has, the length of the axes changes accordingly.
## Matrix

> As we have already described in detail, the matrix is a 2-dimensional tensor.
Tensor
From three or more dimensions one speaks actually also of tensors. A tensor with three dimensions can be thought of as a collection of matrices.

## Different Objects in Python Programming 

Tensors are the generalisation from the objects, which are already known from linear algebra. They are used in programming mainly because they offer the possibility to represent multidimensional data structures.

What arithmetic Operations can be done with them?
The allowed arithmetic operations are very similar to those of matrices, but may differ in naming:

Addition: If two objects have the same dimensions, they can be simply added by adding them element by element, and a new object with the same dimensionality is created.

Subtraction: If two objects have the same dimensions, they can be easily subtracted by subtracting them element by element, resulting in a new object with the same dimensionality.

Hadamard Product: This special product is created by multiplying the objects element by element. The special name comes from the fact that there is still the “normal” way of multiplication, which is based on matrix multiplication.
Division: If two objects have the same dimensions, they can be divided simply by dividing them element by element, and a new object with the same dimensionality is created

What arithmetic Operations can be done with them?
The allowed arithmetic operations are very similar to those of matrices, but may differ in naming:
Addition: If two objects have the same dimensions, they can be simply added by adding them element by element, and a new object with the same dimensionality is created.

Subtraction: If two objects have the same dimensions, they can be easily subtracted by subtracting them element by element, resulting in a new object with the same dimensionality.
Hadamard Product: This special product is created by multiplying the objects element by element. The special name comes from the fact that there is still the “normal” way of multiplication, which is based on matrix multiplication.

Division: If two objects have the same dimensions, they can be divided simply by dividing them element by element, and a new object with the same dimensionality is created.
What is their Function in Machine Learning?

In order to train a machine learning model, a lot of data is needed. However, the data as we know it from the real world is not in the mathematical form in which the model can use it. 
Therefore, we must first convert images, videos, or text, for example, into a multidimensional data structure so that they can be interpreted mathematically.

At the same time, due to their structure, neural networks can accommodate and output a variety of dimensions that go far beyond conventional vectors and matrices. Therefore, with the proliferation of neural networks, the use of tensors has also become more common.
The following applications use these objects in machine learning to increase model performance:
Collaborative Filtering: This model is used in e-commerce, for example, to suggest the most suitable products possible to customers on a website, depending on their previous purchasing behaviour. For training, a matrix was classically used in which the previous customers were displayed as rows and the products as columns. 

If the customer in the first row had already purchased the product in the second column, a 1 was displayed at this point, and if not, a 0. Although this setup was already able to deliver good results, these were further improved by adding dimensions to the matrix, which could then also include the customer context, for example.

Computer Vision: Images and videos cannot be represented as a pure matrix. They not only contain a large number of pixels but are created by superimposing different colours on top of each other, thus creating the image. In simple images in the so-called RGB format, the colours red, green, and blue are superimposed.

## Tensor Representation of an Image 

Notes to Remember about Tensors
Tensors are mathematical objects from linear algebra and are used to represent multidimensional objects.
They can be used to perform the same arithmetic operations that are already familiar with vectors or matrices, for example.
In machine learning, they are used, for example, to achieve better recommendations or to map images and videos in data structures

That means there’s 1 dimension of 3 by 3

Name
What is it?
Number of dimensions
Lower or upper (usually/example)
scalar
a single number
0
Lower (a)
vector
a number with direction (e.g. wind speed with direction) but can also have many other numbers
1
Lower (y)
matrix
a 2-dimensional array of numbers
2
Upper (Q)
tensor
an n-dimensional array of numbers
can be any number, a 0-dimension tensor is a scalar, a 1-dimension tensor is a vector
Upper (X)

# Create a random tensor with a similar shape to an image tensor
random_image_size_tensor = torch.rand(size=(224,224,3)) # height, width, colour channels (R,G,B)
random_image_size_tensor.shape, random_image_size_tensor.ndim



---
layout: default
title: Python Tutorial 2
permalink: /tutorial2/
mathjax: True
---

## Tutorial 2

### Table of contents
- [Introduction](#introduction)
- [Computation Graph](#computation-graph)
- [Introduction to PyTorch](#1-pytorch)
- [Introduction to Tensorflow](#2-tensorflow)
- [References](#3-references)


### Introduction
There are many resources on Machine Learning tools, but this short document aims to condense the most relevant information to help you get started quickly. We recommend exploring your favorite ML framework through the original documentation. Some relevant links can be found in the references.

In machine learning, we often perform numerical operations on data, involving numerous multiplications and additions on real numbers. Multi-dimensional data in a computing language is represented as arrays. In Python, arrays are lists of numbers. Since we perform numerical operations on high-dimensional data, our code needs to be optimized for time and space complexities.

Although the NumPy Python package has almost all functions that work on multidimensional arrays, in machine learning we use a new data representation called a tensor. If you're interested in learning more about tensors, a representation widely popular among physicists, please read the Wikipedia article on [Tensor](https://en.wikipedia.org/wiki/Tensor) or watch the video by [Dr. Daniel A. Fleisch](https://www.youtube.com/watch?v=f5liqUk0ZTw&ab_channel=DanFleisch) for better intuition.

Computer engineers usually keep things simple. The tensors we encounter in machine learning frameworks are simply multi-dimensional arrays optimized to perform gradients quickly, taking advantage of modern GPU architectures. So, next time you see tensors in machine learning frameworks, decouple them from a physicist's idea of a tensor.

PyTorch and TensorFlow are the most popular machine learning frameworks. PyTorch is an open-source, community-driven framework preferred by academics and machine learning enthusiasts. TensorFlow is popular in the industry and can run learning algorithms on edge devices.


### Computation Graph
A graph has nodes and edges. A directed graph has edges marked with directions. Numerical computations in PyTorch and TensorFlow are represented as a directed graph called a computation graph. In a computation graph, nodes represent numerical operations. Below is an example of a computation graph that operates on two inputs. Our neural network architectures are first converted to computation graphs, and then data is passed through.

To understand why we need such a representation, recall the difference between a CPU and a GPU. A CPU has multiple cores, and each core is a processing unit that can strictly perform one task or operation at a time. However, GPUs have hundreds to thousands of tiny cores, and every year, GPU makers race to beat the numbers. Each core on the GPU performs only one numerical operation at a time. The computation graphs take advantage of these tiny cores to run numerical operations simultaneously.

This is a very simplified introduction, but we highly recommend reading more or taking a [parallel computing course](https://gfxcourses.stanford.edu/cs149/fall21/). The following simple recommendation often helps: <i>whenever you write code, think how to parallelize your algorithm.</i>

### 1. PyTorch

PyTorch, developed by the AI research team at Meta, was publicly released in 2016 and is known for its ease of learning. If you prefer to use PyTorch for your work, we recommend the [PyTorch documentation](https://pytorch.org/docs/stable/index.html), which includes blogs, tutorials, and interesting projects on their [GitHub](https://github.com/pytorch/pytorch).

Please play with the code below to initialize tensors, perform mathematical operations, integrate with Numpy arrays, and perform easy gradient calculations. Irrespective of your preferred tool, it's essential to write ML code that takes advantage of a GPU, and we encourage students to spend time becoming comfortable with these tools while learning theoretical ML in the course.

An ideal machine learning researcher has strong programming skills, mathematical knowledge, and good intuition of algorithms to address massive engineering challenges.

### 1.1 Install PyTorch
In Google Colab, Python environments come with Python installed. Please run the following command to check:

```bash
    !pip list | grep torch
```

If you use a preferred local Jupyter notebook, please run a code cell with the below command to install pytorch:

```bash
    !pip install torch
```

### 1.2 Initialization of Tensors
```python
# create a tensor in pytorch
import torch
# create a tensor
x = torch.tensor([5.5, 3]) # 1D tensor
print(x)
# create a 2D tensor
x = torch.tensor([[5.5, 3], [2, 4]])
print(x)
# create a 3D tensor
x = torch.tensor([[[0, 1], [2, 3]], [[5, 6], [7, 8]]])
print(x)
# shape of tensor
print(x.size())
# create a tensor with random values
x = torch.rand(5, 3)
# create a tensor with zeros with specified shape
x = torch.zeros(5, 3, dtype=torch.long)
# print all methods of tensor
print(dir(x))
```

### 1.3 Basic Mathematical Operations
```python
# math operations on tensors
# create tensors
x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])
# addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)
print(z1)
z2 = torch.add(x, y)
print(z2)
# subtraction
z = x - y
# division
z = torch.true_divide(x, y)
# inplace operations
t = torch.zeros(3)
t.add_(x)
t += x  # t = t + x
# exponentiation
z = x.pow(2)
z = x ** 2
# simple comparison
z = x > 0
# matrix multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)  # 2x3
x3 = x1.mm(x2)
# matrix exponentiation
matrix_exp = torch.rand(5, 5)
matrix_exp.matrix_power(3)
# element wise multiplication
z = x * y
print(z)
# dot product
z = torch.dot(x, y)

```

### 1.4 Cloning and Assignment
```python
# tensor assignment vs cloning
import torch
# tensor assignment
x = torch.tensor([1, 2, 3])
y = x
y[0] = -1
print(x)  # tensor([-1,  2,  3])
# tensor cloning
x = torch.tensor([1, 2, 3])
y = x.clone()
y[0] = -1
print(x)  # tensor([1, 2, 3])
```

### 1.5 Auto-differentiation
```python
# tensor auto diffentiation
import torch
# create a simple polynomial function
# f(x) = 3x^2 + 2x - 1
# f'(x) = 6x + 2
# f''(x) = 6
# create a tensor and set requires_grad=True to track computation with it
# this will allow us to compute the gradient of y with respect to x
# use a range of values for x to see how the gradient changes
# for x in range(-10, 11):
#     x = torch.tensor([x], requires_grad=True)
#     y = 3*x**2 + 2*x - 1
#     y.backward()
#     print(x.grad)
x = torch.tensor([2.0], requires_grad=True)
y = 3*x**2 + 2*x - 1
print(y)
# compute the gradient of y with respect to x
y.backward()
print(x.grad)

```

### 2. TensorFlow
TensorFlow creates a static computation graph, whereas Pytorch creates a dynamic computation graph at runtime. When a computation graph is fixed, it helps perform distributed computations effectively. However, it reduces the flexibility of changing things at run-time, and the computation model must be recreated to modify intermediate operations. Researchers have a preference for dynamic computation graphs. We will look at an example comparing static and dynamic graphs in a later tutorial on neural networks.

### 2.1 Basic Mathematical Operations
Tensorflow automatically accelerates numerical operations if a GPU is available. 

```python
# use tensorflow to define tensor and operation
import tensorflow as tf
# create a tensor
hello = tf.constant('Hello, TensorFlow!') # a 0-d tensor
tf.Tensor(1, shape=(), dtype=tf.int32) # a 0-d tensor
tf.Tensor([1, 2, 3], shape=(3,), dtype=tf.int32) # a 1-d tensor
tf.Tensor([[1], [2], [3]], shape=(3, 1), dtype=tf.int32) # a 2-d tensor
# create an operation
a = tf.constant(2)
b = tf.constant(3)
c = a + b
print(c) # tf.Tensor(5, shape=(), dtype=int32)
# run the operation
tf.multiply(a, b) # tf.Tensor(6, shape=(), dtype=int32)
tf.add(a, b) # tf.Tensor(5, shape=(), dtype=int32)
tf.subtract(a, b) # tf.Tensor(-1, shape=(), dtype=int32)
tf.divide(a, b) # tf.Tensor(0.6666666666666666, shape=(), dtype=float64)
c =a**2
print(c) # tf.Tensor(4, shape=(), dtype=int32)
# matrix multiplication
matrix1 = tf.constant([[1, 2], [3, 4]])
matrix2 = tf.constant([[5, 6], [7, 8]])
tf.matmul(matrix1, matrix2) # tf.Tensor([[19, 22], [43, 50]], shape=(2, 2), dtype=int32)
# element-wise multiplication
tf.multiply(matrix1, matrix2) # tf.Tensor([[5, 12], [21, 32]], shape=(2, 2), dtype=int32)
# convert tensor to numpy array and vice versa
# convert tensor to numpy array
import numpy as np
tensor = tf.constant([[1, 2], [3, 4]])
np_array = tensor.numpy()
print(np_array) # [[1 2] [3 4]]
# convert numpy array to tensor
np_array = np.array([[1, 2], [3, 4]])
tensor = tf.constant(np_array)
print(tensor) # tf.Tensor([[1 2] [3 4]], shape=(2, 2), dtype=int64)
# GPU acceleration (if available), tensorflow will automatically use GPU to accelerate the computation
tf.test.is_gpu_available() # True
# use GPU to accelerate the computation
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0], shape=[2, 2], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0], shape=[2, 2], name='b')
    c = tf.matmul(a, b)

```

### 2.2 Gradient Calculations
Gradient calculations are important in machine learning. One should become comfortable calculating gradients.
```python
# automatic gradient calculation using tensorflow
import tensorflow as tf
import numpy as np
# create a variable
x = tf.Variable(3.0)
# define the loss function
with tf.GradientTape() as tape:
    y = x**2
# calculate the gradient
dy_dx = tape.gradient(y, x)
print(dy_dx)
# calculate the gradient of a function
x = tf.Variable(3.0)
with tf.GradientTape(persistent=True) as tape:
    y = x**2
    z = y**2
dz_dx = tape.gradient(z, x)
dy_dx = tape.gradient(y, x)
print(dz_dx, dy_dx)
# Gradient tape?
# it is a context manager that records operations for automatic differentiation
# it records the forward pass and then uses that information to compute the gradients in the backward pass
# what is with tf.GradientTape(persistent=True) as tape?
# it allows you to compute multiple gradients in the same pass
# if you don't use persistent=True, the tape will be destroyed after the first call to gradient()

```

### 2.3 Computation Graph
We have to create computation graphs in TensorFlow as it is not done automatically. A computation graph allows the compiler to parallelize tasks, simplify numerical expressions, and perform under-the-hood optimizations. This greatly reduces the workload of a programmer.
```python
# automatic gradient calculation using tensorflow
import tensorflow as tf
# define function of quadratic form with two variables
def math_function_normal(x,y):
    return 3*x**2 +2*y**2+ 2*x +y- 1
# create a graph for the function
math_function_graph = tf.function(math_function_normal)
# create x and y tensors
x = tf.Variable(1.0)
y = tf.Variable(1.0)
# math_function_normal(x,y) will return the value of the function
math_function_normal(x,y)
# math_function_graph(x,y) 
# to view nodes in the graph
# print(math_function_graph.get_concrete_function(x,y).graph.get_operations())
# to view code in the graph
# print(math_function_graph.get_concrete_function(x,y).graph.as_graph_def())
# observe speedup in graph mode
import time
start_time = time.time()
for i in range(1000):
    math_function_normal(x,y)
end_time = time.time()
print("Normal time:", end_time-start_time)
start_time = time.time()
for i in range(1000):
    math_function_graph(x,y)    
end_time = time.time()
print("Graph time:", end_time-start_time)

```
---
### 3. References
1. [GitHub Copilot](https://github.com/features/copilot)  
2. [Google Colab](https://colab.research.google.com/)  
3. [PyTorch Tutorials](https://pytorch.org/tutorials/)  
4. [TensorFlow Tutorials](https://www.tensorflow.org/guide/)  





<br>
<br>
---
layout: default
title: Python Tutorial 1
permalink: /tutorial1/
mathjax: True
---

## Tutorial 1

### Table of contents
- [Introduction](#introduction)
- [Python Quick Start Guide](#quick-start-guide)
- [Datatypes](#2-data-types-and-precision)
- [Operators](#operators)
- [Loops and Conditions](#4-loops-and-conditions)
- [Functions](#5-functions)
- [Numpy Basics](#6-numpy-package)
- [References](#references)


### Introdution
Python is a powerful and high-level programming language. With its minimal and extensible architecture, Python has evolved into an important tool for researchers, engineers, and programmers. This short document allows you to quickly get started. There are extensive resources to learn the depth of the language.

- [Getting started guide](https://www.python.org/about/) from Python Foundation.
- If you're new to programming, no need to worry! You can get started from this [resource](https://wiki.python.org/moin/BeginnersGuide/NonProgrammers).
- Think you can pace it up? [For programmers transitioning to Python](https://wiki.python.org/moin/BeginnersGuide/Programmers).
- Remember Python evolves at least annually, and it is recommended to keep track of changes once in a while. [Current status and version](https://devguide.python.org/versions/) and [Official documentation](https://devguide.python.org/documentation/).





### 1. Quick Start Guide 
Python comes pre-installed in all modern operating systems. Using Python in a local machine requires a code editor and development environment. Instead, it may be easier to use cloud-based environments like Google Colab.

- Create an account in [Google Colab](https://colab.research.google.com/).
- Create a Jupyter notebook file.
- Remember, when in doubt, just ask a question in the prompt area of Google Colab.


<figure style="center: auto;">
    <img src="/assets/images/colab2.png/" alt=" " style="width: 70%;">
</figure>





### 2. Data Types and Precision
Python supports all basic data types. A unique feature of Python is that there is no need to declare the type of the data.

```python
integer_variable = 2
# Hexadecimal
integer_variable = 0x212
# Binary
integer_variable = 0b10101
float_variable = 5.0
string_variable = "Machine Learning"  # or 'Machine Learning'
```

#### 2.1 Data Precision
```python
integer_variable=0x12
integer_variable=0b10010
float_variable=0.1234
"{:.3f}".format(float_variable) # change precision of float
# "{:b}".format(integer_variable) # change integer to binary
# "{:x}".format(integer_variable) # change integer to hex
# "{:o}".format(integer_variable) # change integer to octal
# "{:d}".format(integer_variable) # change integer to decimal
# "{:c}".format(integer_variable) # change integer to ascii
# "{:e}".format(float_variable) # change float to scientific notation
# "{:E}".format(float_variable) # change float to scientific notation
# "{:f}".format(float_variable) # change float to decimal
# "{:.xf}".format(float_variable) # change float to x decimal places

```  

### 3. Operators
#### 3.1. Arithmetic Operators
```python
x=10
y=20
x+y
print(x+y)
print(x-y)
print(x*y)
print(x/y)  #float division     
print(x//y) #integer division
print(x**y) #power
print(x%y)  #modulus
```
#### 3.2. Relational Operators
```python
x=10
y=20
x+y
print(x>y) # greater
print(x<y) # less   
print(x==y) # equal
print(x!=y) # not equal
print(x>=y) # greater or equal
```
#### 3.3. Logical Operators
```python
x=True 
y=False
x=0
y=1
x>y and y<x
x>y or y<x
not x>y
x=-10 # any number other than zero is True
print(not x)
x=-200 # check what happens
y=1000
print(x and y)
```
#### 3.4. Bitwise Operators
```python
# bitwise operators
a=0b1010
b=0b1100
print("{:b}".format(a&b)) # bitwise and
print("{:b}".format(a|b)) # bitwise or
print("{:b}".format(a^b)) # bitwise xor
print("{:b}".format(~a)) # bitwise not
print("{:b}".format(a<<1)) # bitwise left shift
print("{:b}".format(a>>1)) # bitwise right shift
print("{:b}".format(a>>2)) # bitwise right shift
# observe output of below and try to do circular shift
a=0b1010
b=0b11000
print("{:b}".format(a&b)) # bitwise and
```
#### 3.5. String Operators
Playing with strings highlights the unique advantage of Python over low-level languages. Difficult operations like concatenation, reversal, uppercase transformations, justification, and format specifications are programmer-friendly, easy to memorize, and make it fun to achieve awesome things quickly.
```python
str1='machine learning'
str2=str3[::-1] # string reversal
print(str3)
str1+str2 # string concatenation
str1*3 # string multiplication
str1[0] # string indexing
str1[1:3] # string slicing
str1[-1] # string negative indexing
str1[-3:-1] # string negative slicing
str1[0:3:2] # string slicing with step
str1[::2] # string slicing with step
# illegal string operations
str1[0]='a' # string assignment
str1+1 # string addition
```
#### 3.6. Collections
Python has several data structures to store and handle 
As a language developed with progressive ideas, it can handle a collection of different data types with ease and elegance.
#### 3.6.1 Lists 
List stores collection of same or mixed data types. 
```python
list_of_integers=[1,2,3,4,5,6,7,8,9,10]
list_of_integers_floats=[1.0,2.0,3.0,4.0,5,6.0,7.0,8.0,9,10]
list_of_strings_and_numbers=['1','two','3','four','5','six','7','eight','9','ten']
# indexing lists
print(list_of_integers[0])
print(list_of_integers[-1]) # last element
print(list_of_integers[0:3]) # first 3 elements
print(list_of_integers[3:]) # elements 3 to end
print(list_of_integers[::2]) # every other element
print(list_of_integers[::-1]) # reverse order
print(list_of_integers[1:8:2]) # every other element from 1 to 8
# list comprehensions
squares = [x**2 for x in list_of_integers]
# list comprehensions with conditionals
even_squares = [x**2 for x in list_of_integers if x%2==0]
# list comprehensions with conditionals and else
even_squares = [x**2 if x%2==0 else x**3 for x in list_of_integers]
# list comprehensions with nested loops
pairs = [[x,y] for x in list_of_integers for y in list_of_integers]
# list concatenation
list_of_integers = list_of_integers + list_of_integers
list_of_integers += list_of_integers
```
#### 3.6.2 Dictionaries
Another very powerful data structure that is a fundamental building block of a database is a dictionary. A dictionary stores data in key-value pairs, where the value can be accessed using the key.
```python
# dictionaries hold key value pairs, and ubiquitous in many programming languages, and data bases
# dictionaries are mutable, and can be changed
# key is immutable, value can be mutable
# keys are unique, values need not be unique
# keys are case sensitive, must be immutable, and can be strings, numbers, or tuples
dict = {'name': 'Zach', 'height': 1.8, 'weight': 80, 'BMI': 24.69, 'predicted_weight': 80.5}
# access the value of a key
dict['name']
type(dict['height'])
dict={1:'Zach', 2:'ML', 'third':80, (4,'fourth'):24.69, 5:80.5}
dict[(4,'fourth')]
dict[1]
dict['third']
# concatenate dictionaries
dict1 = {'name': 'Zach', 'height': 1.8, 'weight': 80, 'BMI': 24.69, 'predicted_weight': 80.5}
dict2 = {1:'Zach', 2:'ML', 'third':80, (4,'fourth'):24.69, 5:80.5}
dict1.update(dict2)
```
#### 3.6.3 Tuples
Tuples are similar to lists, however they are immutable (values stored cannot be changed after defining a tuple)
```python
# tuples
tup = (1,2,3)
tup = (1,2,3,4,5,6,7,8,9,10)
tup = (1,2,3,'street', 'city', 'state', 'zip',1.11)
# tuples are similar to lists, but they are immutable
# indexing and slicing
tup[0]
tup[0:3]
tup[3:6]
# tuples are immutable
# tup[0] = 100 # this will throw an error
```

### 4 Loops and Conditions
#### 4.1 Loops
Loops help in navigating data structures and are useful in writing algorithms.  
```python
# for loop
list_of_integers_floats_strings = [1, 2.0, '3']
# simple for loop
for item in list_of_integers_floats_strings:
    print(item)
# for loop with enumerate
for index, item in enumerate(list_of_integers_floats_strings):
    print(index, item)
# enumerate returns a tuple of the index and the item
# list comprehension
[item for item in list_of_integers_floats_strings]
# list comprehension with enumerate
[index for index, item in enumerate(list_of_integers_floats_strings)]
# list comprehension with nested for loops
# helps in creating a list of lists
[[item for item in list_of_integers_floats_strings] for index, item in enumerate(list_of_integers_floats_strings)]
```
#### 4.2 Conditions
 ```python
 conditional statements in python
# if, elif, else
# if condition:
#     do something
# elif condition:
#     do something
# else:
#     do something
 ```











### 5. Functions

We can write reusable snippets of code using functions. Python supports both built-in and user-defined functions.

#### 5.1 List of built-in functions

Knowledge of built-in functions comes in handy to write code quickly, improves code readability, and of course saves a lot of time.
<style>
 
  table {
    width: 95%;
    background-color: #e8edf1; /* Background color added here */
    margin-left: auto; /* Center table with automatic left margin */
    margin-right: auto; /* Center table with automatic right margin */
    margin-top: 10px; /* Optional: Adds top margin for spacing */
    margin-bottom: 10px; /* Optional: Adds bottom margin for spacing */
  }
  th, td {
    padding: 8px;
  }
  th:first-child, td:first-child {
    padding-right: 20px; /* Increase right padding of the first column */
  }
  td:nth-child(2) {
    background-color: #d6eaff; /* New background color for the second column */
  }
</style>

<table > 
  <thead>
    <tr> 
      <th style="background-color:#e8edf1;"><strong>Function</strong></th>
      <th style="background-color:#d6eaff;"><strong>Description</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>abs()</td>
      <td>Returns the absolute value of a number.</td>
    </tr>
    <tr>
      <td>aiter()</td>
      <td>Returns an asynchronous iterator for an asynchronous iterable.</td>
    </tr>
    <tr>
      <td>all()</td>
      <td>Returns True if all elements of an iterable are true.</td>
    </tr>
    <tr>
      <td>anext()</td>
      <td>Asynchronously returns the next item from an asynchronous iterator.</td>
    </tr>
    <tr>
      <td>any()</td>
      <td>Returns True if any element of an iterable is true.</td>
    </tr>
    <tr>
      <td>ascii()</td>
      <td>Returns a string containing a printable representation of an object.</td>
    </tr>
    <tr>
      <td>bin()</td>
      <td>Converts an integer to a binary string.</td>
    </tr>
    <tr>
      <td>bool()</td>
      <td>Converts a value to a Boolean.</td>
    </tr>
    <tr>
      <td>breakpoint()</td>
      <td>Calls the built-in breakpoint function for debugging.</td>
    </tr>
    <tr>
      <td>bytearray()</td>
      <td>Returns a bytearray object.</td>
    </tr>
    <tr>
      <td>bytes()</td>
      <td>Returns a bytes object.</td>
    </tr>
    <tr>
      <td>callable()</td>
      <td>Returns True if the object appears callable.</td>
    </tr>
    <tr>
      <td>chr()</td>
      <td>Returns a Unicode character string with a specified code point.</td>
    </tr>
    <tr>
      <td>classmethod()</td>
      <td>Returns a class method for a function.</td>
    </tr>
    <tr>
      <td>compile()</td>
      <td>Compiles the source into a code or AST object.</td>
    </tr>
    <tr>
      <td>complex()</td>
      <td>Returns a complex number.</td>
    </tr>
    <tr>
      <td>delattr()</td>
      <td>Deletes an attribute from an object.</td>
    </tr>
    <tr>
      <td>dict()</td>
      <td>Returns a new dictionary.</td>
    </tr>
    <tr>
      <td>dir()</td>
      <td>Returns a list of names in the current local scope or a list of attributes of an object.</td>
    </tr>
    <tr>
      <td>divmod()</td>
      <td>Returns the quotient and the remainder when dividing two numbers.</td>
    </tr>
    <tr>
      <td>enumerate()</td>
      <td>Returns an enumerate object.</td>
    </tr>
    <tr>
      <td>eval()</td>
      <td>Evaluates a string as a Python expression.</td>
    </tr>
    <tr>
      <td>exec()</td>
      <td>Executes a dynamically created Python program.</td>
    </tr>
    <tr>
      <td>filter()</td>
      <td>Returns an iterator from elements of an iterable for which a function returns true.</td>
    </tr>
    <tr>
      <td>float()</td>
      <td>Returns a floating-point number from a number or a string.</td>
    </tr>
    <tr>
      <td>format()</td>
      <td>Formats a specified value.</td>
    </tr>
    <tr>
      <td>frozenset()</td>
      <td>Returns an immutable frozenset object.</td>
    </tr>
    <tr>
      <td>getattr()</td>
      <td>Returns the value of a named attribute of an object.</td>
    </tr>
    <tr>
      <td>globals()</td>
      <td>Returns a dictionary representing the current global symbol table.</td>
    </tr>
    <tr>
      <td>hasattr()</td>
      <td>Returns True if an object has the given named attribute.</td>
    </tr>
    <tr>
      <td>hash()</td>
      <td>Returns the hash value of an object.</td>
    </tr>
    <tr>
      <td>help()</td>
      <td>Displays help related to a specific object or module.</td>
    </tr>
    <tr>
      <td>hex()</td>
      <td>Converts an integer to a lowercase hexadecimal string.</td>
    </tr>
    <tr>
      <td>id()</td>
      <td>Returns the identity of an object.</td>
    </tr>
    <tr>
      <td>input()</td>
      <td>Reads a line from the standard input.</td>
    </tr>
    <tr>
      <td>int()</td>
      <td>Returns an integer from a number or a string.</td>
    </tr>
    <tr>
      <td>isinstance()</td>
      <td>Returns True if an object is an instance of a specified type.</td>
    </tr>
    <tr>
      <td>issubclass()</td>
      <td>Returns True if a class is a subclass of a specified class.</td>
    </tr>
    <tr>
      <td>iter()</td>
      <td>Returns an iterator object.</td>
    </tr>
    <tr>
      <td>len()</td>
      <td>Returns the length (the number of items) of an object.</td>
    </tr>
    <tr>
      <td>list()</td>
      <td>Returns a list.</td>
    </tr>
    <tr>
      <td>locals()</td>
      <td>Updates and returns a dictionary representing the current local symbol table.</td>
    </tr>
    <tr>
      <td>map()</td>
      <td>Applies a function to all items in an input list and returns an iterator.</td>
    </tr>
    <tr>
      <td>max()</td>
      <td>Returns the largest item in an iterable or the largest of two or more arguments.</td>
    </tr>
    <tr>
      <td>memoryview()</td>
      <td>Returns a memory view object.</td>
    </tr>
    <tr>
      <td>min()</td>
      <td>Returns the smallest item in an iterable or the smallest of two or more arguments.</td>
    </tr>
    <tr>
      <td>next()</td>
      <td>Retrieves the next item from an iterator by calling its <code>__next__()</code> method.</td>
    </tr>
    <tr>
      <td>object()</td>
      <td>Returns a new featureless object.</td>
    </tr>
    <tr>
      <td>oct()</td>
      <td>Converts an integer to an octal string.</td>
    </tr>
    <tr>
      <td>open()</td>
      <td>Opens a file and returns a file object.</td>
    </tr>
    <tr>
      <td>ord()</td>
      <td>Returns an integer representing the Unicode character.</td>
    </tr>
    <tr>
      <td>pow()</td>
      <td>Returns x to the power of y, with an optional z as a modulus.</td>
    </tr>
    <tr>
      <td>print()</td>
      <td>Prints the specified message to the screen.</td>
    </tr>
    <tr>
      <td>property()</td>
      <td>Gets, sets, or deletes a property of an object.</td>
    </tr>
    <tr>
      <td>range()</td>
      <td>Returns a sequence of numbers.</td>
    </tr>
    <tr>
      <td>repr()</td>
      <td>Returns a string containing a printable representation of an object.</td>
    </tr>
    <tr>
      <td>reversed()</td>
      <td>Returns a reversed iterator of a sequence.</td>
    </tr>
    <tr>
      <td>round()</td>
      <td>Rounds a number to the nearest integer or to the specified number of decimals.</td>
    </tr>
    <tr>
      <td>set()</td>
      <td>Returns a new set or converts an iterable to a set.</td>
    </tr>
    <tr>
      <td>setattr()</td>
      <td>Sets the value of a named attribute of an object.</td>
    </tr>
    <tr>
      <td>slice()</td>
      <td>Returns a slice object.</td>
    </tr>
    <tr>
      <td>sorted()</td>
      <td>Returns

 a sorted list from the specified iterable.</td>
    </tr>
    <tr>
      <td>staticmethod()</td>
      <td>Returns a static method for a function.</td>
    </tr>
    <tr>
      <td>str()</td>
      <td>Returns a string version of an object.</td>
    </tr>
    <tr>
      <td>sum()</td>
      <td>Returns the sum of all items in an iterable.</td>
    </tr>
    <tr>
      <td>super()</td>
      <td>Returns a temporary object that allows access to a parent class.</td>
    </tr>
    <tr>
      <td>tuple()</td>
      <td>Returns a tuple.</td>
    </tr>
    <tr>
      <td>type()</td>
      <td>Returns the type of an object.</td>
    </tr>
    <tr>
      <td>vars()</td>
      <td>Returns the <code>__dict__</code> attribute for a module, class, instance, or any other object with a <code>__dict__</code> attribute.</td>
    </tr>
    <tr>
      <td>zip()</td>
      <td>Returns an iterator of tuples, where the first item in each passed iterator is paired together, and so on.</td>
    </tr>
    <tr>
      <td><code>__import__()</code></td>
      <td>Imports a module by name.</td>
    </tr>
  </tbody>
</table>



#### 5.2 User Defined Functions
We can define functions with and without arguments. 
```python
# Syntax for function definition
def my_function():
        # write logic here
    print('Hello World')
# Syntax for function call
my_function()
```

### 6. Numpy Package
Python packages help us perform quick execution of desired operations. The Numpy package provides a comprehensive collection of functions for numerical operations on arrays. Please find the following resources for more information:

- [Documentation](https://numpy.org/doc/stable/user/basics.html)
- [Basic Usage](https://numpy.org/doc/stable/user/basics.html)
- [Tutorials](https://numpy.org/numpy-tutorials/index.html)

```python
# Syntax for function definition
# import numpy module
import numpy as <any_name, say np>
import numpy as np
# list of numpy functions
dir(np)
# help on a function
help(np.array)
# use a function
np.array([1,2,3]) # returns an array of 1,2,3
# data types in numpy
# int, float, bool, str, object, complex, bytes, unicode
```

---

### References
1. Github copilot. URL: https://github.com/features/copilot.
2. Google colab. URL: https://colab.research.google.com/.
3. Numpy documentation. URL: https://numpy.org/doc/stable/index.
html.
4. Python documentation. URL: https://devguide.python.org/
documentation/.
5. Python for non-programmers. URL: https://wiki.python.org/moin/
BeginnersGuide/NonProgrammers.
6. Python for programmers. URL: https://wiki.python.org/moin/
BeginnersGuide/Programmers.
7. Python foundation. URL: https://www.python.org/abou a Boolean.   
<br>
<br>
---
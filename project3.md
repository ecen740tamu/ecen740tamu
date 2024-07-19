---
layout: default
title: Convolutional Neural Networks
permalink: /project3/
mathjax: True
---


## Implement Transformer with Self-Attention


### Introduction
Over the past few decades, significant efforts have been made to understand language. Language, as a primary means of human communication, reflects our culture, evolution, and facilitates connections between individuals. Linguistics formally studies the fundamental structure of languages and investigates whether there exists any common ground across different languages.  It is remarkable to witness a computer algorithm learning the intricacies of various languages and comprehending the context of a sentence. However, it's important to recognize that this portrayal of an algorithm's capabilities may be somewhat exaggerated.

 Language models are continuously evolving, and there is a substantial amount of human effort involved in their development, as you will discover while completing this project. One of the most compelling aspects of the <strong><i>**attention mechanism**</i></strong> is its ability to learn and accurately preserve the context in both input and output. 
The work of Vaswani et al., <strong><i>**Attention is All You Need**,</i></strong> NeurIPS 2017, significantly advanced the state of the art in language modeling.
It showcased the ability of transformers not only for translation. Interestingly, it has been quite challenging for me to find an area or task transformers are not applicable. Also, BERT is encoder only model and GPT models are decoder only.

### Task
- Implement English-to-German translation task.

 - Get the dataset from [Huggingface](https://huggingface.co/datasets/wmt/wmt14).

- Since the dataset is large (4.5 million entries), please use just the first 80,000 translations. 
  Feel free to use more rows in the dataset if you have access to the appropriate computing resources. 
There might be a significant need for computing power to finish the task with more entries. 

- Reserve 0.1$\%$ of the 80,000, i.e., 80 entries, for testing.
- Use either byte pair encoding or word-level tokenization. Feel free to try others as well.

- Please use exactly all the parameters as described in the paper, except for the number of entries in the dataset. (This is to make sure the code runs smoothly).
    
- Run for 100 epochs and the rest of the architecture of the paper must remain the same. 

- Save the trained model after all epochs have been completed, to save disk space.
- Use the attached validation.py file to run validations on the final trained model. The validation.py file contains two small translation datasets. 
   (Table 1 display some of the entries in the validation set.) Please print a table with the predicted translation produced by your implementation, and the correct translation for the two datasets. 


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
<table>
  <caption>Table 1: German Sentences and their English Translations</caption>
  <thead>
    <tr>
      <th>German</th>
      <th>English</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Guten Morgen!</td>
      <td>Good morning!</td>
    </tr>
    <tr>
      <td>Wie geht es dir?</td>
      <td>How are you?</td>
    </tr>
    <tr>
      <td>Ich bin hungrig.</td>
      <td>I am hungry.</td>
    </tr>
    <tr>
      <td>Entschuldigung, wo ist die Toilette?</td>
      <td>Excuse me, where is the restroom?</td>
    </tr>
    <tr>
      <td>Wie viel kostet das?</td>
      <td>How much does that cost?</td>
    </tr>
    <tr>
      <td>Ich spreche kein Deutsch.</td>
      <td>I don't speak German.</td>
    </tr>
    <tr>
      <td>Was ist dein Name?</td>
      <td>What is your name?</td>
    </tr>
    <tr>
      <td>Es tut mir leid.</td>
      <td>I'm sorry.</td>
    </tr>
    <tr>
      <td>Woher kommst du?</td>
      <td>Where are you from?</td>
    </tr>
    <tr>
      <td>Ich liebe dich.</td>
      <td>I love you.</td>
    </tr>
    <tr>
      <td>Wie spät ist es?</td>
      <td>What time is it?</td>
    </tr>
    <tr>
      <td>Kannst du mir helfen?</td>
      <td>Can you help me?</td>
    </tr>
    <tr>
      <td>Ich verstehe nicht.</td>
      <td>I don't understand.</td>
    </tr>
    <tr>
      <td>Auf Wiedersehen!</td>
      <td>Goodbye!</td>
    </tr>
    <tr>
      <td>Wo ist der Bahnhof?</td>
      <td>Where is the train station?</td>
    </tr>
    <tr>
      <td>Ich habe eine Frage.</td>
      <td>I have a question.</td>
    </tr>
    <tr>
      <td>Wie alt bist du?</td>
      <td>How old are you?</td>
    </tr>
    <tr>
      <td>Ich bin müde.</td>
      <td>I am tired.</td>
    </tr>
    <tr>
      <td>Was machst du gerne in deiner Freizeit?</td>
      <td>What do you like to do in your free time?</td>
    </tr>
    <tr>
      <td>Was ist das?</td>
      <td>What is that?</td>
    </tr>
    <tr>
      <td>Mein Name ist John.</td>
      <td>My name is John.</td>
    </tr>
    <tr>
      <td>Wie heißt das auf Deutsch/Englisch?</td>
      <td>What is that called in German/English?</td>
    </tr>
    <tr>
      <td>Ich bin beschäftigt.</td>
      <td>I am busy.</td>
    </tr>
    <tr>
      <td>Wie war dein Tag?</td>
      <td>How was your day?</td>
    </tr>
    <tr>
      <td>Ich habe Hunger.</td>
      <td>I am hungry.</td>
    </tr>
  </tbody>
</table>


<br>
<br>

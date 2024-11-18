# Mad Libs

## Project Description

This project leverages Google's BERT, a transformer-based language model, to predict a masked word based on the context of a piece of writing. Masked Language Models like BERT are the foundation for LLMs, today's groundbreaking technology. This program predicts the most probable words for a masked term and visualizes the model's attention to the surrounding context, illustrating how BERT prioritizes information for precise predictions.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Technologies](#technologies)
- [Credit](#credit)

## Installation

You'll need to have Python and pip3 installed. You can download them from the [official Python website](https://www.python.org/downloads/).

1. Clone the repository:

```bash
git clone https://github.com/ColinDao/mad-libs.git
```

2. Navigate to the project directory:

```bash
cd mad-libs
```

3. Install the required dependencies:

```bash
pip3 install -r requirements.txt
```

## Usage

To start the program, run the following command:

```bash
python mask.py
```

Then, enter a piece of writing with a word replaced with "[MASK]". This will be what BERT predicts! Look at its suggestions and how the model's attention changes for each thing it reads.

## Features

**Masked Language Modeling**: Utilize Google’s BERT, a transformer-based model, to predict masked words in a sentence by understanding and leveraging context from surrounding words.  <br />
<br />
**Text Preprocessing**: Preprocess input text by tokenizing it using BERT’s tokenizer, which splits text into subwords and maps them to the model’s vocabulary, ensuring compatibility with the BERT architecture and preserving linguistic nuance.<br />
<br />
**Prediction Generation**: Use BERT to identify the top candidates for the masked word, outputting these as possible predictions based on the context provided by the input sentence. <br />
<br />
**Attention Visualization**: Generate visualizations that map BERT’s self-attention scores, highlighting how the model interprets and attends to each token's context within a sentence. These visualizations allow for a deeper understanding of BERT’s contextual decision-making process.

## Technologies
**Language**: Python <br />
**Libraries**: Transformers, NumPy, Pillow, Sys, TensorFlow

## Credit

This project was completed as a part of [CS50's Introduction to Artificial Intelligence with Python](https://cs50.harvard.edu/ai/2024/). Go check them out!

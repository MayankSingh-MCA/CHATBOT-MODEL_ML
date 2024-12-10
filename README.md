# CHATBOT-MODEL_ML
A machine learning chatbot that automates customer support queries using Naive Bayes and TF-IDF. It provides instant, accurate responses for inquiries like order status, refunds, and delivery times. Built with Python and Flask API, it supports integration with web apps and platforms like WhatsApp or Messenger.
# Customer Support Chatbot

## Overview

The **Customer Support Chatbot** is a machine learning-based solution designed to automate responses to common customer queries, such as order status, delivery times, refunds, and cancellations. The chatbot utilizes a **Naive Bayes classifier** combined with **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization to classify user intents and provide relevant, context-specific responses.

Built with **Python**, **Scikit-learn**, and **Flask**, the system can be deployed as an API for real-time interactions. The chatbot aims to enhance customer experience by providing instant responses, reducing the workload of customer support teams, and improving operational efficiency.

## Features

- **Automated Query Resolution**: Handle repetitive customer queries instantly.
- **Accurate Intent Classification**: Classify user queries with **95%** accuracy using Naive Bayes.
- **Instant Responses**: Provide real-time responses to common customer inquiries.
- **Scalable Design**: Easily expandable to handle additional intents and queries.
- **Flask-based Deployment**: Host the chatbot using Flask API for easy integration with web platforms.
- **Customizable**: Easily add new intents and responses to suit your business needs.
- **Future Enhancements**: Plans to integrate advanced NLP models (e.g., BERT), multi-language support, and messaging platform integration (e.g., WhatsApp, Slack).

## Technologies Used

- **Python**: Programming language used for the chatbot development.
- **Scikit-learn**: Machine learning library used for training the Naive Bayes classifier.
- **TF-IDF**: Text vectorization technique used to convert text data into numerical features.
- **Flask**: Framework for deploying the chatbot as an API.
- **Joblib**: For saving and loading the trained machine learning model.
- **NLTK**: Natural Language Toolkit for text preprocessing (optional for tokenization).

## Getting Started

### Prerequisites

To run this project, you will need the following:

- Python 3.x
- Git (for version control)
- Dependencies (listed in requirements.txt)

### Installation

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/Customer-Support-Chatbot.git
cd Customer-Support-Chatbot
pip install -r requirements.txt
python app.py

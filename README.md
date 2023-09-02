# AWS Documentation POC

## Overview

This repository contains the code for the tech assessment from Loka for the Senior ML Engineer position. A simple description of the task is that Company X has a lot of in house technical documentation for which its developers spend a great amount of time navigate through or ask other developers for pointers. The request is to find a solution to shorten the time spent on documentation and also on taking up time of more experienced developers whose work is interupted to help answer questions. The dataset to build is AWS documentation which is publicly available. Below are the main painpoints to solve and the addition features that later on the POC would be extended to:

| Main Pain-Point (POC) | Nice-Haves (Final System)      |
| ---------------| -----------------|
| Reduce the time developers spend on navigating through internal documentation. | Point to the source of the response.   |
|  | Suggest documents for further reading.   |
|  | Final system will need to handle internal documents that have proprietory and geographical restrictions.|

## Approach

The above problem description can be boiled down to the POC having to a couple of capabilities. Firstly the system should be able to gather AWS documentation and then be able to digest AWS documentation. Secondly, there has to be an ML/LLM model that is finetuned on the domain specific data. And finally, the system wil have to be able to actually answer questions such as "What is Amazon SageMaker?".

I initially did extensive research on current approaches or even any SOTA. In the <a href="https://paperswithcode.com/dataset/aws-documentation">paperswithcode.com</a> there is a dataset and 2 research papers but both are for some proprietary software and do not provide exact code.

There are a couple of approaches when it comes to solving this problem. One approach could be the development of an Information Retrievel and Question Answering System (such as described in this git repo <a href="https://github.com/spyros-briakos/Document-Retrieval-and-Question-Answering-with-BERT/blob/main/README.md">Q&A</a>) on some LLM (such as BERT, GPT-3.5, Alpacca, LLaMa, Vicuna). This approch is benefitial since it will solve the querying of docuemnts and is also very easily extendible to solving the additional problem of Pointing to the source and Suggesting sources for further reading. In this repository I did not choose this approach since it requires extensive resources and a well refined dataset. There are wrappers and third party libraries and tools that make it eassily yo train such models such as <a href="https://python.langchain.com/docs/use_cases/question_answering/">LangChain</a> however i am unable to utilize this since it required an API key from OpenAI which is payed for. The other open source LLMs LLaMa, Alpacca, Vicuna are quite large and so finetuning them would take quite some time and resource.

Since the Main Painpoint of this POC is to determine if an ML system can be devloped for Question Answering. I decided to go with a much simpler approach that is not resource intensive and will be able to be "cooked up" quite quickly. I went with a Q&A system based on BERT. The main idea is to take a pretrained BERT model then do Intermidiet Pre-Training on the custom domain specific docuemntation and finally Fine-Tune the model for context Question and Answering. 

Having said all that this is the approach and system capabilites in a nutshell:

- Make a Mechanism for Gather AWS Documentation
- Develop e System for Question Answering based on BERT
- Choose a pretrained BERT
- Intermidiet Pre-Training on the custom domain specific docuemntation
- Fine-Tune the model for context Question and Answering

---
NOTE: **Gathering the AWS documentation** had a couple of challenges. They have git repositories for each service which contains markdown files. This is quite cumbersome to gather and process all but will most probably be the best choice. 

There were some git hub repos which attemted to solve the problem but <a href="https://github.com/richarvey/getAWSdocs">dont really work</a> or have <a href="https://github.com/siagholami/aws-documentation">old and incomplete data</a>

Which is why I chose to scrape the aws pages. To run the scraping you will need a webdriver since i am scraping with a Selenium.

## Table of Contents

- [Project Directory Structure](#project-directory-structure)
- [Requirements](#requirements)
- [How to Run](#How-to-run)
- [Data](#data)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Improvements](#future-improvements)

## Project Directory Structure

Folders:
- **BTF** - contains the Tokenizer
- **data** - contains the sample **dataset.csv** of AWS pages with their title, url and content. This dataset will be used for the Vocabulary Extraction (Step 2), Tokenizer (Step 3) model and will produce the train and validation folders containing the chunked datasets for the fune tuning in Step 4. **SQUAD_Format_Q&A** is a subfolder containing question and answer datasets on AWS services for the final step of Question Answering (Step 5).
- **Dataset_Generation** - contains the script for scraping AWS documentation.
- **finetuned** - contains the files for the finetuned model needed by the transformer library.
- **Q&A** - contains code for just the QA section of the training. Offers a nice islated folder to just play around with the final training task.
- **SageMaker_Step4** - this folder contains a mock up version of the Step4 where we finetune the model. It has not been tested or debugged its just my shot at trying to show a an AWS implementation of a solution. Since i was unable to make the entire system in AWS.

Files:
- **Step 1 - PrepareDataset.ipynb** - contains the code for preparing the dataset and generating the dataset.csv file.
- **Step 2 - ExtractVocabulary.ipynb** - contains code for extracting the vocabulary with the BertWordPieceTokenizer.
- **Step 3 - MLMPreprocessCustom.ipynb** - contains the code for bringing the data into chunks in the approapriate format for the Masked Language Model. And generating the train/validation folders and the dataset_dict.json file.
- **Step 4 - FineTune.ipynb** - This is where we do the Intermidiet Pre-Training on the custom domain specific docuemntation. We see that the model has a **perplexity of 35577.18** and a **perplexity of 10688.52** with only **5 epochs of training**. Perplexity is used in LLMS to show the degree of uncertainty of predicting a sample. In our case used to evaluate how well the context is captured. $$
PPL(W) = \left( P(w_1, w_2, \ldots, w_N) \right)^{-\frac{1}{N}}
$$

- **Step 5 - Q&A.ipynb** - contains the code for training a Q&A model with our custom dataset in the SQUAD format through the transformers library (Huggingface).

## Requirements

- chrome webdriver
- pytorch
- transformers


## How to run

Run the notebooks in consecutive fashion as the names suggest.

## Data

**Gathering the AWS documentation** had a couple of challenges. They have git repositories for each service which contains markdown files. This is quite cumbersome to gather and process all but will most probably be the best choice. 

There were some git hub repos which attemted to solve the problem but <a href="https://github.com/richarvey/getAWSdocs">dont really work</a> or have <a href="https://github.com/siagholami/aws-documentation">old and incomplete data</a>

Which is why I chose to scrape the aws pages. To run the scraping you will need a webdriver since i am scraping with a Selenium. For more documentation just supply more links in the pages.txt file.

## Results

```
from transformers import pipeline

question_answerer = pipeline("question-answering", model=model, tokenizer=tokenizer)


context = """
Amazon SageMaker is a fully managed service that provides tools for building, training, and deploying machine learning models. 
It offers a complete set of machine learning algorithms, integrated development environments, and infrastructure to accelerate the machine learning workflow.
"""
question = "What is Amazon Sagemaker?"
question_answerer(question=question, context=context)
```

```
{'score': 0.04191909730434418,
 'start': 50,
 'end': 137,
 'answer': 'provides tools for building, training, and deploying machine learning models. It offers'}
``````

## Conclusion

As the above sections shows in the end we are able to supply a context on a certain service (SageMaker) and then be able to ask a question and get an accurate response back.

## Future Improvements

There are a handful of improvements for the future
- Implements an IR and Q&A system with more sophisticated LLMs (GPT-4, Alpacca, LLaMA)
- Set up a system to track the official documentation from AWS on their git hub repos. and create a much better documentation dataset
- Set up experiment tracking for trainig through MLflow, wandb, comet.ml, neptune ...
- Set up Approapriate logging for the entire system
- Set up a front end user interface such as -> https://docsgpt.antimetal.com/

---
> :warning: **Test and implements the entire ML pipelines in AWS Sagemaker.**

---
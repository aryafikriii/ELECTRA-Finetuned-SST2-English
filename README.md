# Electra Finetuned SST2 English

This project focuses on fine-tuning and transfer learning of the Google Electra transformer model using the SST-2 dataset for text classification tasks. The goal is to leverage the power of pre-trained models and adapt them to the specific task of sentiment analysis on the SST-2 dataset. The project utilizes the PyTorch framework and runs on a GPU device for efficient training and inference.

## Dataset

The SST-2 dataset consists of labeled sentences from movie reviews, where each sentence is classified as either positive or negative sentiment. The dataset is widely used for sentiment analysis tasks and provides a suitable benchmark for evaluating the performance of the fine-tuned Electra model.

## Model Architecture

The Electra model, developed by Google, is a powerful transformer-based architecture that has shown state-of-the-art performance on various natural language processing (NLP) tasks. By fine-tuning the Electra model on the SST-2 dataset, we aim to create a robust and accurate sentiment analysis model.

## Tech Stack

The Electra Finetuned SST2 English project utilizes the following technologies:

- PyTorch: A deep learning framework used for building and training neural networks.
- GPU: The project leverages GPU acceleration for faster training and inference.

## Project Setup

To set up the project for training and inference, please follow these steps:

1. Clone the project repository.

```
git clone https://github.com/aryafikriii/ELECTRA-Finetuned-SST2-English
```

2. Install the required dependencies by running the following command.

```
pip install -r requirements.txt
```

3. Download the SST-2 dataset and place it in the designated data directory.

4. Run the training and evaluation script to fine-tune the Electra model with full layer and evaluate its performance on the test set.

```
python transferLearning_All-Layer.py
```

Note: Make sure you have a compatible GPU device with the necessary drivers and CUDA toolkit installed to run the training and inference processes efficiently.

## Contact

If you have any questions or suggestions regarding this project, please feel free to contact me. You can reach me at [aryafikriansyah@gmail.com].

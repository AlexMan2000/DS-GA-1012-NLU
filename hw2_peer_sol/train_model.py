"""
Code for Problem 1 of HW 2.
"""
import pickle
from typing import Any, Dict

import evaluate
import numpy as np
import optuna
from datasets import Dataset, load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments, EvalPrediction

def compute_metrics(eval_preds):
    # print('eval_preds',eval_preds)
    accuracy = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(references=labels, predictions=predictions)

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return metric.compute(predictions=predictions, references=labels)


def preprocess_dataset(dataset: Dataset, tokenizer: BertTokenizerFast) \
        -> Dataset:
    """
    Problem 1d: Implement this function.

    Preprocesses a dataset using a Hugging Face Tokenizer and prepares
    it for use in a Hugging Face Trainer.

    :param dataset: A dataset
    :param tokenizer: A tokenizer
    :return: The dataset, prepreprocessed using the tokenizer
    """
    # raise NotImplementedError("Problem 1d has not been completed yet!")
    def tokenize_function(dataset):
        return tokenizer(dataset["text"], padding="max_length", truncation=True, max_length=512)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    return tokenized_datasets


def init_model(trial: Any, model_name: str, use_bitfit: bool = False) -> \
        BertForSequenceClassification:
    """
    Problem 2a: Implement this function.

    This function should be passed to your Trainer's model_init keyword
    argument. It will be used by the Trainer to initialize a new model
    for each hyperparameter tuning trial. Your implementation of this
    function should support training with BitFit by freezing all non-
    bias parameters of the initialized model.

    :param trial: This parameter is required by the Trainer, but it will
        not be used for this problem. Please ignore it
    :param model_name: The identifier listed in the Hugging Face Model
        Hub for the pre-trained model that will be loaded
    :param use_bitfit: If True, then all parameters will be frozen other
        than bias terms
    :return: A newly initialized pre-trained Transformer classifier
    """
    # raise NotImplementedError("Problem 2a has not been completed yet!")
    model = BertForSequenceClassification.from_pretrained(model_name)
    if use_bitfit == True:
        for name, param in model.named_parameters():
            if 'bias' not in name:
                param.requires_grad = False
    return model


def init_trainer(model_name: str, train_data: Dataset, val_data: Dataset,
                 use_bitfit: bool = False) -> Trainer:
    """
    Prolem 2b: Implement this function.

    Creates a Trainer object that will be used to fine-tune a BERT-tiny
    model on the IMDb dataset. The Trainer should fulfill the criteria
    listed in the problem set.

    :param model_name: The identifier listed in the Hugging Face Model
        Hub for the pre-trained model that will be fine-tuned
    :param train_data: The training data used to fine-tune the model
    :param val_data: The validation data used for hyperparameter tuning
    :param use_bitfit: If True, then all parameters will be frozen other
        than bias terms
    :return: A Trainer used for training
    """
    # raise NotImplementedError("Problem 2b has not been completed yet!")
    # Define training arguments
    def initialize_model():
        return init_model(None,model_name,use_bitfit)
    # model = init_model(None,model_name,use_bitfit)
    # print('model',model)

    training_args = TrainingArguments(
    output_dir="./checkpoints",  # Directory to save the checkpoints
    # per_device_train_batch_size=8,
    eval_steps=200,
    save_steps=200,
    num_train_epochs=4,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,  # Save the model at the end of each epoch
)

    trainer = Trainer(
    args=training_args,
    model_init= initialize_model,
    train_dataset=train_data,
    eval_dataset=val_data,
    # device="cuda",
    compute_metrics=compute_metrics)
    return trainer


def hyperparameter_search_settings() -> Dict[str, Any]:
    """
    Problem 2c: Implement this function.

    Returns keyword arguments passed to Trainer.hyperparameter_search.
    Your hyperparameter search must satisfy the criteria listed in the
    problem set.

    :return: Keyword arguments for Trainer.hyperparameter_search
    """
    # raise NotImplementedError("Problem 2c has not been completed yet!")
    # trial = optuna.create_trial()  # Create a dummy trial object
    def hp_space(trial): 
        return {
       "learning_rate": trial.suggest_categorical("learning_rate", [3e-4, 1e-4, 5e-5, 3e-5]),
        # "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        # "seed": trial.suggest_int("seed", 1, 40),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [ 8, 16, 32, 64, 128]),
    }

    search_space = {
        "learning_rate": [ 3e-4, 1e-4, 5e-5, 3e-5],
        # "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 5),
        # "seed": trial.suggest_int("seed", 1, 40),
        "per_device_train_batch_size": [ 8, 16, 32, 64, 128],
    }

    sampler = optuna.samplers.GridSampler(search_space)

    # compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(axis=1) == p.label_ids).mean()}

    return {'hp_space':hp_space, 'direction':'maximize',
    # 'compute_objective':compute_metrics,
    'backend':'optuna','sampler':sampler}






if __name__ == "__main__":  # Use this script to train your model
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset and create validation split
    imdb = load_dataset("imdb")
    split = imdb["train"].train_test_split(.2, seed=3463)
    imdb["train"] = split["train"]
    imdb["val"] = split["test"]
    del imdb["unsupervised"]
    del imdb["test"]

    # Preprocess the dataset for the trainer
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    imdb["train"] = preprocess_dataset(imdb["train"], tokenizer)
    imdb["val"] = preprocess_dataset(imdb["val"], tokenizer)

    # Set up trainer
    trainer = init_trainer(model_name, imdb["train"], imdb["val"],
                           use_bitfit=True)

    # Train and save the best hyperparameters
    # print('use_bitfit:',use_bitfit)
    best = trainer.hyperparameter_search(**hyperparameter_search_settings())
    with open("train_results.p", "wb") as f:
        pickle.dump(best, f)

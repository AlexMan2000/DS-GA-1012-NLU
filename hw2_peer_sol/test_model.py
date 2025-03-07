"""
Code for Problem 1 of HW 2.
"""
import pickle

import evaluate
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments

from train_model import preprocess_dataset
# from datasets import load_metric
import numpy as np

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


def init_tester(directory: str) -> Trainer:
    """
    Prolem 2b: Implement this function.

    Creates a Trainer object that will be used to test a fine-tuned
    model on the IMDb test set. The Trainer should fulfill the criteria
    listed in the problem set.

    :param directory: The directory where the model being tested is
        saved
    :return: A Trainer used for testing
    """
    # raise NotImplementedError("Problem 2b has not been completed yet!")
    model = BertForSequenceClassification.from_pretrained(directory)
    training_args = TrainingArguments(
    # output_dir="./checkpoints",  # Directory to save the checkpoints
    # per_device_train_batch_size=8,
    # num_train_epochs=4,
    output_dir="./results"
    # logging_dir='./logs',
    # save_strategy="epoch",  # Save the model at the end of each epoch
)

    trainer = Trainer(
    args=training_args,
    model= model,
    # train_dataset=train_data,
    # eval_dataset=val_data,
    compute_metrics= compute_metrics,)
    return trainer


if __name__ == "__main__":  # Use this script to test your model
    model_name = "prajjwal1/bert-tiny"

    # Load IMDb dataset
    imdb = load_dataset("imdb")
    del imdb["train"]
    del imdb["unsupervised"]

    # Preprocess the dataset for the tester
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    imdb["test"] = preprocess_dataset(imdb["test"], tokenizer)

    # Set up tester
    path_to_your_best_model = "checkpoints/run-5/checkpoint-628"
    tester = init_tester(path_to_your_best_model)

    # Test
    results = tester.predict(imdb["test"])
    with open("test_results.p", "wb") as f:
        pickle.dump(results, f)

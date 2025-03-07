"""
Code for Problem 1 of HW 2.
"""
import pickle

import evaluate
from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments

from train_model import preprocess_dataset
import numpy as np
def compute_metrics(eval_preds):
    # print('eval_preds',eval_preds)
    accuracy = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(references=labels, predictions=predictions)



def init_tester(directory: str) -> Trainer:
    """
    Problem 2b: Implement this function.

    Creates a Trainer object that will be used to test a fine-tuned
    model on the IMDb test set. The Trainer should fulfill the criteria
    listed in the problem set.

    :param directory: The directory where the model being tested is saved.
    :return: A Trainer used for testing.
    """
    model = BertForSequenceClassification.from_pretrained(directory)
    training_args = TrainingArguments(
        output_dir=directory,  # Set output directory for test results
        per_device_eval_batch_size=8,  # Batch size for testing
        do_predict=True,  # Ensures the Trainer is used for inference
        logging_dir=f"{directory}/logs",  # Log directory
    )

    trainer = Trainer(
        args=training_args,
        model=model,
        compute_metrics=compute_metrics  # Function for evaluation
    )

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
    tester = init_tester("./checkpoints-test")

    # Test
    results = tester.predict(imdb["test"])
    with open("test_results.p", "wb") as f:
        pickle.dump(results, f)

# Homework 2 Written Question Answers

## Problem 1a

- **Figure 2** is for the **main experiment** answering "yes" to "larger models are less truthful" and **figure 4** shows the results from **additional experiment** that explores the average truthfulness/informativeness of different models with different number of parameters.
- There are 5 sets of prompts in section E of rthe appendix, which are "QA", "Harmful", "Helpful", "chat", "long-form". The main experiments use "QA" since it only tests for truthfulness of the LLM QA performance. The additional experiment uses all five sets of prompts since it not only tests the truthfulness, but the informativeness of the LLM in generaration task.

## Problem 1b
- Method 1 is **generation** in which the model is given a prompt and question and the model generates the answer to the question. Method 2 is **multiple choice** in which the model is given a prompt, a question and a set of reference answers. The model will assign likelihood to each of the answers.
- For method 1, we use human evaluation to score models on truthfulness and informativeness, where a model’s score is the percentage of its responses that a human judges to be true or informative. For method 2, it is calculated by the total
normalized likelihood of the true answers (normalized across all true and false reference answers).


## Problem 1c
- **Difference between MC1 and MC2**: In MC1, there is only one correct answer out of 4-5 reference answers but in MC2 there are multiple correct answers.
- **Difference between MC1 and text classification**: MC1's candidate answers don't necessarily have a fixed list of categories while text-classification usually only has two categories(positive/negative). Moreover, the correct answers in MC1 should be factual and truthlly grounded while labels in sentimental analysis may be biased across different human interpretations. Most importantly, MC1 is in essence a comparison-based evaluation by comparing the log probabiilties that model assign to each candidate answers while text classification is a decision task by evaluating whether model can correctly map input text to a class label.


## Problem 3a


## Problem 3b



## Problem 3c

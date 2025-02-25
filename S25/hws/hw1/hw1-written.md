# Homework 1 Written Question Answers

## Problem 1c
Because in out __getitem__ method, we are expecting an Iterable[str] as input. If we input a list of string like in the first use case ["the", "of"], it will index into the embedding array and fetch the embeddings for the words "the" and "of" respectively. Otherwise, if we input a single string "the" as in the second use case, it will be treated as ["t", "h", "e"] where these characters don't have their corresponding embedding in the embedding array.


## Problem 4a


## Problem 4b


## Problem 4c


# Homework 1 Written Question Answers

## Problem 1c
Because in out __getitem__ method, we are expecting an Iterable[str] as input. If we input a list of string like in the first use case ["the", "of"], it will index into the embedding array and fetch the embeddings for the words "the" and "of" respectively. Otherwise, if we input a single string "the" as in the second use case, it will be treated as ["t", "h", "e"] where these characters don't have their corresponding embedding in the embedding array.


## Problem 4a
All the accuracy results for the experiement are as follows:
| Embedding Space | Semantic | Syntactic | Overall |
|-----------------|----------|-----------|---------|
| GloVe 50        | 0.384    |    0.245       |    0.295    |
| GloVe 100       |    0.413      |    0.247       |    0.307    |
| GloVe 200       |    0.313      |    0.184       |    0.230    |


From the results obtained from our experiment and the original paper, we see that the Glove Embedding's **overall** performance on the analogy tests are close to that of CBOW while significantly worse than the Skip-Gram model. More specifically, in the semantic analogy test, the Glove embedding performs better than CBOW. In the syntactic analogy test, the Glove embedding performs worse than both word2vec models. 

The dimensionality of the embedding space does not have a significant impact on the performance of the Glove embedding. At least there is no positive or negative correlation between the performance of the Glove embedding and the dimensionality of the embedding space.

## Problem 4b
All the accuracy results for the experiement are as follows:
| Embedding Space | Semantic | Syntactic | Overall |
|-----------------|----------|-----------|---------|
| GloVe 50        | 0.560      |    0.504       |    0.524    |
| GloVe 100       |    0.634      |    0.628       |    0.630    |
| GloVe 200       |    0.658      |    0.634       |    0.642    |

Now the results are significantly better than the previous results. The performance of the Glove embedding is now better than that of the Skip-Gram model with large embedding size. Also there seems to be a positive correlation between the performance of the Glove embedding and the dimensionality of the embedding space. 



## Problem 4c
The results for the experiment are as follows:

| Analogy Question                | Gold Answer  | GloVe 50 | GloVe 100 | GloVe 200 |
|---------------------------------|--------------|----------|-----------|-----------|
| france : paris :: italy : _x_   | rome         |    rome      |     rome      |      rome     |
| france : paris :: japan : _x_   | tokyo        |   tokyo      |     tokyo     |      tokyo    |
| france : paris :: florida : _x_ | tallahassee  |  miami  |   florida     |      florida   |
| big : bigger :: small : _x_     | smaller      |  larger     |     larger   |      smaller  |
| big : bigger :: cold : _x_      | colder       |  cold     |     cold   |      cold  |
| big : bigger :: quick : _x_     | quicker      |  quick    |     quick   |      quick  |

In this experiment, we see that the 
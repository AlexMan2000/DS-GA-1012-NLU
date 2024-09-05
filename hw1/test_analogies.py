"""
Code for Problems 2 and 3 of HW 1.
"""
from typing import Dict, List, Tuple

import numpy as np

from embeddings import Embeddings


def cosine_sim(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Problem 3b: Implement this function.

    Computes the cosine similarity between two matrices of row vectors.

    :param x: A 2D array of shape (m, embedding_size)
    :param y: A 2D array of shape (n, embedding_size)
    :return: An array of shape (m, n), where the entry in row i and
        column j is the cosine similarity between x[i] and y[j]
    """
    
    x = np.apply_along_axis(lambda row_vec: row_vec / np.sqrt(row_vec @ row_vec), 1, x)
    y = np.apply_along_axis(lambda row_vec: row_vec / np.sqrt(row_vec @ row_vec), 1, y)

    return x @ y.T

def get_closest_words(embeddings: Embeddings, vectors: np.ndarray,
                      k: int = 1) -> List[List[str]]:
    """
    Problem 3c: Implement this function.

    Finds the top k words whose embeddings are closest to a given vector
    in terms of cosine similarity.

    :param embeddings: A set of word embeddings
    :param vectors: A 2D array of shape (m, embedding_size)
    :param k: The number of closest words to find for each vector
    :return: A list of m lists of words, where the ith list contains the
        k words that are closest to vectors[i] in the embedding space,
        not necessarily in order
    """

    index_to_words = {i: w for i, w in enumerate(embeddings.words)}

    def find_closest(vec):
        return list(map(lambda i: index_to_words[i], np.argsort(cosine_sim(embeddings.vectors, vec.reshape(1, -1)).reshape(-1))[::-1][:k]))

    return list(map(find_closest, vectors))
    


# This type alias represents the format that the testing data should be
# deserialized into. An analogy is a tuple of 4 strings, and an
# AnalogiesDataset is a dict that maps a relation type to the list of
# analogies under that relation type.
AnalogiesDataset = Dict[str, List[Tuple[str, str, str, str]]]


def load_analogies(filename: str) -> AnalogiesDataset:
    """
    Problem 2b: Implement this function.

    Loads testing data for 3CosAdd from a .txt file and deserializes it
    into the AnalogiesData format.

    :param filename: The name of the file containing the testing data
    :return: An AnalogiesDataset containing the data in the file. The
        format of the data is described in the problem set and in the
        docstring for the AnalogiesDataset type alias
    """
    analogy_dict = {}
    key = None
    tuple_list = []

    with open(filename, "r") as analogy_file:
        
        for line in analogy_file:
            line = line.strip()
            if line.startswith(":"):
                if len(tuple_list) != 0:
                    analogy_dict[key] = tuple_list
                    tuple_list = []
                key = line.split(" ")[1]
                analogy_dict[key] = None
                continue
            tuple_list.append(tuple(line.split(" ")))
        # The last one
        analogy_dict[key] = tuple_list

    return analogy_dict



def run_analogy_test(embeddings: Embeddings, test_data: AnalogiesDataset,
                     k: int = 1) -> Dict[str, float]:
    """
    Problem 3d: Implement this function.

    Runs the 3CosAdd test for a set of embeddings and analogy questions.

    :param embeddings: The embeddings to be evaluated using 3CosAdd
    :param test_data: The set of analogies with which to compute analogy
        question accuracy
    :param k: The "lenience" with which accuracy will be computed. An
        analogy question will be considered to have been answered
        correctly as long as the target word is within the top k closest
        words to the target vector, even if it is not the closest word
    :return: The results of the 3CosAdd test, in the format of a dict
        that maps each relation type to the analogy question accuracy
        attained by embeddings on analogies from that relation type
    """

    return 

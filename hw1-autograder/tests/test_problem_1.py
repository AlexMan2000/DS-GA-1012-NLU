import random
import unittest
from typing import Dict

import numpy as np

from _utils import try_function, try_loading_student_embedding
from embeddings import Embeddings as StuEmbeddings
from solution.embeddings import Embeddings as SolEmbeddings


class TestProblem1(unittest.TestCase):
    """
    This test case tests the Embeddings functions on a number of
    randomly generated word embedding files.
    """

    def __init__(self, *args, **kwargs):
        super(TestProblem1, self).__init__(*args, **kwargs)
        self.stu_embeddings_cache: Dict[int, StuEmbeddings] = dict()
        self.sol_embeddings_cache = \
            [SolEmbeddings.from_file(self._get_filename(i)) for i in range(5)]

    @staticmethod
    def _get_filename(index: int) -> str:
        return "data/sample_embeddings_{}.txt".format(index)

    def get_student_embedding(self, i: int, try_init: bool = True) -> \
            StuEmbeddings:
        """
        Tries to load an embeddings file using student code.

        :param i: Loads the file sample_embeddings_i.txt
        :param try_init: If True, the code will try loading with
            __init__ if from_file fails
        :return: The embeddings file, if successfully loaded; otherwise,
            the exception that was raised
        """
        if i not in self.stu_embeddings_cache:
            f = self._get_filename(i)
            print(f'Loading embeddings using embeddings = '
                  f'Embeddings.from_file("{f}")...')

            # Try to load the embeddings
            sol_embeddings = self.sol_embeddings_cache[i]
            words = sol_embeddings.words if try_init else None
            vecs = sol_embeddings.vectors if try_init else None
            stu_embeddings = try_loading_student_embedding(f, words, vecs)

            # Cache the embeddings
            self.stu_embeddings_cache[i] = stu_embeddings
            return stu_embeddings
        else:
            return self.stu_embeddings_cache[i]

    def test_load_from_file(self):
        """Problem 1b: Test Embeddings.from_file."""
        for i in range(5):
            # Preliminary message for the student
            f = self._get_filename(i)
            print(f"Testing on a sample embeddings file called {f}.")

            # Load solution embeddings
            sol_embeddings = self.sol_embeddings_cache[i]

            # Try to load from file
            if i in self.stu_embeddings_cache:
                del self.stu_embeddings_cache[i]
            stu_embeddings = self.get_student_embedding(i, try_init=False)

            # Check dimensionality of embeddings
            stu_shape = stu_embeddings.vectors.shape
            sol_shape = sol_embeddings.vectors.shape
            assert stu_shape == sol_shape, \
                (f"embeddings.vectors has shape {stu_shape}, when it should "
                 f"be {sol_shape}")

            # Check embeddings
            stu_vectors = stu_embeddings.vectors
            sol_vectors = sol_embeddings.vectors
            assert np.isclose(stu_vectors, sol_vectors).all(), \
                "embeddings.vectors has the wrong values."

            # Check number of words
            stu_vocab_size = len(stu_embeddings.words)
            sol_vocab_size = len(sol_embeddings)
            assert stu_vocab_size == sol_vocab_size, \
                (f"{f} has a vocab size of {sol_vocab_size}, but only "
                 f"{stu_vocab_size} words have been loaded into "
                 f"embeddings.words.")

            # Check membership of words
            stu_words = stu_embeddings.words
            sol_words = sol_embeddings.words
            for j in range(sol_vocab_size):
                assert stu_words[j] == sol_words[j], \
                    (f"Word {j} of the embeddings file is {sol_words[j]}, "
                     f"but word {j} of embeddings.words is {stu_words[j]}.")

            print(f"Test passed for {f}!\n")

    def test_len(self):
        """Problem 1b: Test Embeddings.__len__."""
        for i in range(5):
            # Preliminary message for the student
            f = self._get_filename(i)
            print(f"Testing on a sample embeddings file called {f}.")

            # Try to load the embeddings
            stu_embeddings = self.get_student_embedding(i)

            # Try to call __len__
            with try_function("len(embeddings)"):
                stu_len = len(stu_embeddings)

            # Check that the length is correct
            words_len = len(stu_embeddings.words)
            vecs_len = len(stu_embeddings.vectors)
            assert stu_len in [words_len, vecs_len], \
                (f"The vocab size of embeddings is {vecs_len}, but len("
                 f"embeddings) returned {stu_len}")

            print(f"Test passed for {f}!\n")

    def test_contains(self):
        """Problem 1b: Test Embeddings.__contains__."""
        with open("data/glove_vocab_500.txt", "r") as f:
            glove_vocab = [line.strip() for line in f]

        for i in range(5):
            # Preliminary message for the student
            f = self._get_filename(i)
            print(f"Testing on a sample embeddings file called {f}.")

            # Try to load embeddings
            stu_embeddings = self.get_student_embedding(i)

            for w in glove_vocab:
                if w in stu_embeddings.words:
                    assert w in stu_embeddings, \
                        (f'"{w}" in embeddings returns False even though '
                         f"embeddings.words contains {w}")
                else:
                    assert w not in stu_embeddings, \
                        (f'"{w}" in embeddings returns True even though '
                         f"embeddings.words does not contain {w}")

            print(f"Test passed for {f}!\n")

    def test_getitem(self):
        """Problem 1b: Test Embeddings.__getitem__."""
        for i in range(5):
            # Preliminary message for the student
            f = self._get_filename(i)
            print(f"Testing on a sample embeddings file called {f}.")

            # Try to load the embeddings
            stu_embeddings = self.get_student_embedding(i)

            for j in range(5):
                # Choose some random words from the vocab
                words = random.choices(stu_embeddings.words,
                                       k=random.randint(1, 10))
                idx = [stu_embeddings.words.index(w) for w in words]
                sol_vectors = stu_embeddings.vectors[idx]

                # Try calling __contains__
                with try_function(f"embeddings[{words}]"):
                    stu_vectors = stu_embeddings[words]

                # Check number of embeddings
                assert len(stu_vectors) == len(words), \
                    (f"embeddings[{words}] returned {len(stu_vectors)} "
                     f"vectors, but it should have returned {len(words)}.")

                # Check value of embeddings
                assert (sol_vectors == stu_vectors).all(), \
                    f"embeddings[{words}] returns incorrect vectors."

            print(f"Test passed for {f}!\n")

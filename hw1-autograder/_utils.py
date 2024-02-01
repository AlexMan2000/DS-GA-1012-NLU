from contextlib import contextmanager
from typing import List, Optional, Tuple

import numpy as np

from embeddings import Embeddings as StuEmbeddings


@contextmanager
def try_function(function_name: str):
    """
    This is a custom try-except block, with more helpful error messages.
    Since students do not have access to the autograder, pure Python
    error messages will be uninterpretable to them. With this context
    manager, you can use the function_name parameter to describe what
    part of the code is being tested when an error is raised.
    """
    try:
        yield
    except Exception as e:
        msg = f"A(n) {type(e).__name__} occurred while calling " \
              f"{function_name}. Error message: {e}"
        raise e.__class__(msg) from e


def try_loading_student_embedding(
        filename: str, words: Optional[List[str]] = None,
        vecs: Optional[np.ndarray] = None) -> StuEmbeddings:
    """
    Tries to load student embeddings using .from_file. If this fails,
    then tries to construct student embeddings using .__init__.

    :param filename: The file to load from
    :param words: The words, in case .from_files fails
    :param vecs: The vectors, in case .from_files fails
    :return: The student embeddings
    """
    try_init = words is not None and vecs is not None
    from_file_failed = False

    try:
        with try_function(f'Embeddings.from_file("{filename}")'):
            stu_embeddings = StuEmbeddings.from_file(filename)
    except Exception as e:
        if not try_init:
            raise e
        from_file_failed = True
        print("Embeddings.from_file failed. See the from_file unit "
              "test for more details.\nManually creating embeddings "
              "object using Embeddings.__init__...")

        with try_function("Embeddings.__init__"):
            stu_embeddings = StuEmbeddings(words, vecs)

    constructor = "Embeddings.__init__" if from_file_failed else \
        "Embeddings.from_file"
    assert isinstance(stu_embeddings, StuEmbeddings), \
        (f"{constructor} returned an object of type "
         f"{type(stu_embeddings).__name__} instead of an Embeddings "
         f"object.")

    return stu_embeddings


if __name__ == "__main__":
    with try_function("some part of the code") as t:
        raise RuntimeError("The code raised a RuntimeError")

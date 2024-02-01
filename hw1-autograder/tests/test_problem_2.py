import unittest

from _utils import try_function
from solution.test_analogies import load_analogies as sol_load_analogies
from test_analogies import load_analogies as stu_load_analogies


class TestProblem2(unittest.TestCase):
    """
    This test case tests the load_analogies on a randomly generated
    analogies file.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.longMessage = False

    def test_load_analogies(self):
        """Problem 2b: Test load_analogies."""
        f = "data/sample_analogies.txt"
        print(f"Testing load_analogies on a file called {f}.\n"
              f'Calling analogies = load_analogies("{f}")...')

        # Try loading student analogies
        with try_function(f'load_analogies("{f}")'):
            stu_analogies = stu_load_analogies(f)
        sol_analogies = sol_load_analogies(f)

        # Check if it's a dict
        assert isinstance(stu_analogies, dict), \
            "load_analogies is supposed to return a dict, but it returns a " \
            f"{type(stu_analogies).__name__} instead."

        # Check relation types
        assert len(stu_analogies) == len(sol_analogies), \
            (f"{f} contains {len(sol_analogies)} relation types, but only "
             f"{len(stu_analogies)} have been loaded.")

        self.assertListEqual(
            sorted(stu_analogies.keys()), sorted(sol_analogies.keys()),
            msg="The relation types loaded by load_analogies do not match "
                f"those in {f}.")

        for r_type in sol_analogies:
            with try_function(f'"{r_type}" in analogies'):
                assert r_type in stu_analogies, \
                    f"analogies is missing a relation type from {f}"

        for r_type in stu_analogies:
            assert r_type in sol_analogies, \
                f"analogies has an extra relation type not in {f}"

        # Check analogies (can assume that all relation types are there)
        for r_type in stu_analogies:
            stu_n = len(stu_analogies[r_type])
            sol_n = len(sol_analogies[r_type])
            assert stu_n == sol_n, \
                (f'analogies["{r_type}"] contains {stu_n} analogies, but {f} '
                 f'contains {sol_n} analogies for relation type {r_type}.')

            self.assertListEqual(
                sorted(stu_analogies[r_type]), sorted(sol_analogies[r_type]),
                msg=f'The analogies in analogies["{r_type}"] do not match '
                    f'those in {f} for relation type {r_type}.')

        print("Test passed!")

import unittest  # The test framework

from Lab1 import *  # The code to test


class Test_TestIncrementDecrement(unittest.TestCase):
    def test_increment(self):
        self.assertEqual(inc_dec.increment(3), 4)

    def test_decrement(self):
        self.assertEqual(inc_dec.decrement(3), 2)


if __name__ == "__main__":
    unittest.main()

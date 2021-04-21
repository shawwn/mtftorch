from mtftorch.testing._internal.common_utils import TestCase, run_tests

class TestBasic(TestCase):
    def test_basic(self):
        self.assertEqual(True, True)

if __name__ == '__main__':
    run_tests()

from mtftorch.testing._internal.common_utils import TestCase, run_tests

from mtftorch import nn

class MyModule(nn.Module):
    def __init__(self, foo=42, bar=99):
        super().__init__()
        self.foo = 42
        self.bar = 99

    def forward(self, x):
        return x + 1

    def extra_repr(self):
        return 'foo={foo}, bar={bar}' \
               ''.format(**self.__dict__)


class TestModule(TestCase):
    def setUp(self):
        super().setUp()
        self.module = MyModule()

    def test_module_forward(self):
        self.assertEqual(2, self.module(1))

    def test_module_extra_repr(self):
        self.assertEqual('MyModule(foo=42, bar=99)', str(self.module))

if __name__ == '__main__':
    run_tests()

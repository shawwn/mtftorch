from mtftorch.testing._internal.common_utils import TestCase, run_tests

import mtftorch
from mtftorch import mtf
from mtftorch import nn

class TestModule(TestCase):
    def setUp(self):
        super().setUp()
        self.identity = nn.Identity()

    def test_identity(self):
        self.assertEqual(2, self.identity(1) + 1)

    def test_reprs(self):
        self.assertEqual(f'Identity()', str(self.identity))

    def test_view(self):
        image_nchw = mtf.zeros(mtftorch.get_mesh(), mtftorch.shapelist("N=1 C=3 H=4 W=4"))
        image_nhwc = image_nchw.view("N H W C")
        self.assertEqual(image_nchw.shape.size, image_nhwc.shape.size)
        self.assertNotEqual(image_nchw.size(-1), image_nhwc.size(-1))
        self.assertEqual(image_nchw.size(0), image_nhwc.size(0))
        for dim in 'N C H W'.split():
            self.assertEqual(image_nchw.size(dim), image_nhwc.size(dim))


if __name__ == '__main__':
    run_tests()

from mtftorch.testing._internal.common_utils import TestCase, run_tests

import mtftorch as torch
from mtftorch import tf, nn


class TestModule(TestCase):
    def setUp(self):
        super().setUp()

    def test_identity(self):
        identity = nn.Identity()
        self.assertEqual(2, identity(1) + 1)
        self.assertEqual(f'Identity()', str(identity))

    def test_permute(self):
        channel = torch.ones("H=4 W=4")
        pixel = torch.tensor([0.1, 0.5, 0.9], "C")
        image_chw = pixel * channel
        self.assertEqual(
            image_chw,
            torch.stack([0.1 * channel, 0.5 * channel, 0.9 * channel], "C"),
        )
        image_nchw = image_chw["N", torch.newaxis]
        self.assertEqual(
            image_nchw,
            torch.stack([image_chw], "N")
        )
        image_nhwc = image_nchw.permute("N H W C")
        self.assertEqual(
            image_nchw.shape.size,
            image_nhwc.shape.size)
        self.assertNotEqual(
            image_nchw.size(-1),
            image_nhwc.size(-1))
        self.assertEqual(
            image_nchw.size(0),
            image_nhwc.size(0))
        self.assertEqual(
            torch.tensor([[0.1, 0.5, 0.9]], "N C"),
            image_nchw.mean("H W"))
        self.assertEqual(
            torch.tensor([[16*0.1, 16*0.5, 16*0.9]], "N C"),
            image_nchw.sum("H W"))
        for dim in "N C H W".split():
            self.assertEqual(image_nchw.size(dim), image_nhwc.size(dim))
        self.assertEqual(
            torch.tensor([0.1, 0.5, 0.9], "C"),
            image_nchw.index("W", 0).index("H", 0).index("N", 0))
        self.assertEqual(
            torch.tensor([0.1, 0.5, 0.9], "C"),
            image_nchw["W", 0, "H", 0, "N", 0])
        self.assertEqual(
            image_nhwc.view("N HW=-1 C").shape,
            torch.size("N=1 HW=16 C=3")
        )

    def test_rand(self):
        image1 = torch.randint(10, "    H=4 W=4 C=3", seed=(2, 3)).float()
        image2 = torch.randint(10, "N=1 H=4 W=4 C=3", seed=(2, 3)).float()
        self.assertEqual(image1, image2.squeeze("N"))

    def test_rand_and_slicing(self):
        image = torch.randint(10, "N=1 H=4 W=4 C=3", seed=(2, 3)).float()
        image = image.squeeze("N")
        image = image.permute("H W C")
        tile_top, tile_bot = image.split("H", 2)
        tile_top: torch.TensorType
        tile_bot: torch.TensorType
        tile_00, tile_10 = tile_top.split("W", 2)
        tile_00: torch.TensorType
        tile_10: torch.TensorType
        tile_01, tile_11 = tile_bot.split("W", 2)
        tile_01: torch.TensorType
        tile_11: torch.TensorType
        tiles = [tile_00, tile_10, tile_01, tile_11]
        self.assertEqual(
            tiles[0].shape,
            torch.size("H=2 W=2 C=3"))
        rgbs = [x.unbind("C") for x in tiles]
        R, G, B = torch.mtf.transpose_list_of_lists(rgbs)
        R: torch.List[torch.TensorType]
        G: torch.List[torch.TensorType]
        B: torch.List[torch.TensorType]
        self.assertEqual(
            R[0].shape,
            G[0].shape)
        self.assertEqual(
            R[0].shape,
            torch.size("H=2 W=2"))
        tile_ul = image["H", 0:2, "W", 0:2]
        self.assertEqual(
            tile_ul,
            tile_00)
        tile_ul_rgb = tile_ul.unbind("C")
        # import pdb; pdb.set_trace()
        self.assertEqual(
            tile_ul_rgb[0],
            R[0])

    def test_import_tf_tensor(self):
        a = torch.tensor(tf.random.stateless_uniform([2, 2, 3], seed=(2, 3)), "H=2 W=2 C=3")
        b = torch.tensor(tf.random.stateless_uniform([2, 2, 3], seed=(2, 3)), "H   W   C  ")
        self.assertEqual(a, b)


if __name__ == '__main__':
    run_tests()

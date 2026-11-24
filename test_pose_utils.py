import unittest

from pose_utils import reached_chest_height


class TestReachedChestHeight(unittest.TestCase):
    def test_kick_above_chest(self):
        self.assertTrue(reached_chest_height(0.35, 0.5))

    def test_kick_below_chest(self):
        self.assertFalse(reached_chest_height(0.65, 0.5))

    def test_kick_at_chest_height(self):
        self.assertTrue(reached_chest_height(0.5, 0.5))


if __name__ == "__main__":
    unittest.main()

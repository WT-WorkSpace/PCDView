import os
import tempfile
import unittest

import numpy as np

from utils.slam import load_pose_file, pose_values_to_matrix, transform_xyz


class SlamPoseTest(unittest.TestCase):
    def test_xyz_rpy_and_transform(self):
        pose = pose_values_to_matrix([1, 2, 3, 0, 0, np.pi / 2])
        point = transform_xyz(np.array([[1, 0, 0, 42]], dtype=float), pose)
        np.testing.assert_allclose(point, [[1, 3, 3]], atol=1e-7)

    def test_quaternion_with_timestamp(self):
        pose = pose_values_to_matrix([123.0, 1, 2, 3, 0, 0, 0, 1])
        np.testing.assert_allclose(pose[:3, 3], [1, 2, 3])
        np.testing.assert_allclose(pose[:3, :3], np.eye(3))

    def test_matrix_and_comments(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("# frame poses\n")
                f.write("000001.pcd 1 0 0 4 0 1 0 5 0 0 1 6 # matrix\n")
            poses = load_pose_file(path)
            self.assertEqual(len(poses), 1)
            np.testing.assert_allclose(poses[0][:3, 3], [4, 5, 6])
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()

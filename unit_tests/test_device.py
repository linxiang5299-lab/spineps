# Call 'python -m unittest' on this folder
from __future__ import annotations

import unittest
from unittest import mock

import torch

from spineps.utils.device import pick_device, pick_device_name


def _patch(cuda: bool, mps: bool):
    return (
        mock.patch("torch.cuda.is_available", return_value=cuda),
        mock.patch("torch.backends.mps.is_available", return_value=mps),
    )


class Test_Device_Selection(unittest.TestCase):
    def test_use_cpu_wins_over_accelerators(self):
        p_cuda, p_mps = _patch(cuda=True, mps=True)
        with p_cuda, p_mps:
            self.assertEqual(pick_device_name(use_cpu=True), "cpu")
            self.assertEqual(pick_device(use_cpu=True), torch.device("cpu"))

    def test_cuda_is_preferred(self):
        p_cuda, p_mps = _patch(cuda=True, mps=True)
        with p_cuda, p_mps:
            self.assertEqual(pick_device_name(), "cuda")
            self.assertEqual(pick_device(), torch.device("cuda:0"))

    def test_mps_when_no_cuda(self):
        p_cuda, p_mps = _patch(cuda=False, mps=True)
        with p_cuda, p_mps:
            self.assertEqual(pick_device_name(), "mps")
            self.assertEqual(pick_device(), torch.device("mps"))

    def test_cpu_when_nothing_is_available(self):
        p_cuda, p_mps = _patch(cuda=False, mps=False)
        with p_cuda, p_mps:
            self.assertEqual(pick_device_name(), "cpu")
            self.assertEqual(pick_device(), torch.device("cpu"))

    def test_missing_mps_backend_attribute(self):
        with mock.patch("torch.cuda.is_available", return_value=False), mock.patch.object(torch.backends, "mps", None, create=True):
            self.assertEqual(pick_device_name(), "cpu")


if __name__ == "__main__":
    unittest.main()

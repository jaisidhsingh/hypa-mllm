import torch
from absl.testing import absltest
from absl.testing import parameterized

from src.model.mllm import MLLM


class TestingMLLM(parameterized.TestCase):
    @classmethod
    def __init__(self):
        self.batch_size = 4
        self.image_size = 224
        self.seq_len = 128

    @torch.no_grad()    
    def test_initialisation(self):
        pass

    @torch.no_grad()
    def test_modality_connector(self):
        pass

    @torch.no_grad()
    def test_only_llm(self):
        pass

    @torch.no_grad()
    def test_only_vision_tower(self):
        pass

    @torch.no_grad()
    def test_full_mllm(self):
        pass

    @torch.no_grad()
    def test_hypernetwork(self):
        pass

    @torch.no_grad()
    def test_full_hypermllm(self):
        pass


def run_model_unit_tests():
    absltest.main(__name__)

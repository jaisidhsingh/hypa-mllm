import torch
from absl.testing import absltest
from absl.testing import parameterized

from src.model.mllm import MLLM
from src.model.hypernetworks import HyperNetwork


# class TestingMLLM(parameterized.TestCase):
class TestingMLLM():
    # def setUp(self):
    def __init__(self):
        self.batch_size = 4
        self.image_size = 224
        self.num_patches = 197
        self.dim = 768
        self.seq_len = 128
        self.llm = "llama-3.2"
        self.vision_tower = "vanilla_vit_b16"
        self.connector_type = "linear"
        self.modules_to_freeze = ["vision_tower", "llm"]
        self.device = "cpu"
        self.model = MLLM(
            llm=self.llm,
            vision_tower=self.vision_tower,
            connector_type=self.connector_type,
            connector_hidden_dims=[],
            modules_to_freeze=self.modules_to_freeze,
            device=self.device
        )

    @torch.no_grad()
    def test_mllm_forward_pass(self):
        forward_pass_correctly_done = False
        try:
            images = torch.randn((1, 3, self.image_size, self.image_size))
            input_text = "A cat sat on the"
            input_text_enc = self.model.tokenizer(input_text, return_tensors="pt")
            input_text_ids = input_text_enc.input_ids

            target_text = " mat."
            target_text_enc = self.model.tokenizer(target_text, return_tensors="pt")
            target_text_ids = target_text_enc.input_ids

            full_text = input_text + target_text
            full_text_enc = self.model.tokenizer(full_text, return_tensors="pt")
            full_text_ids = full_text_enc.input_ids
            attention_mask = full_text_enc.attention_mask
            
            labels = torch.full((1, self.model.num_patches+full_text_ids.shape[1]), fill_value=-100, dtype=torch.long)
            labels[:, self.model.num_patches+input_text_ids.shape[1]:] = target_text_ids

            output = self.model(full_text_ids, images, attention_mask, labels)

            print("Forward pass correctly executed.")
            forward_pass_correctly_done = True

            print("Outputs obtained are:")
            print(output.keys())

        except Exception as err:
            print("Error in forward pass.")
            print(err)

        # self.assertEqual(forward_pass_correctly_done, True)
        
    @torch.no_grad()
    def test_hypernetwork(self):
        hypnet_works = False
        # try:
        param_shapes = [[768, 768], [768]]
        hypnet = HyperNetwork(
            param_shapes=param_shapes,
            cond_emb_dim=8,
            num_cond_embs=3,
            image_embed_dims=[768],
            hidden_layer_factors=[]
        )
        print("Hypernetwork initisialised correctly.")

        weights, biases = hypnet([0, 1, 2], 768)
        hypnet_works = True

        print("Hypernetwork forward pass correctly executed.")
        print("Obtained params for connects have dims:")
        print(weights.shape, biases.shape)
        
        # except:
        #     print("Error in hypernetwork.")
        
        # self.assertEqual(hypnet_works, True)

    @torch.no_grad()
    def test_full_hypermllm(self):
        """
        Not implemented yet.
        """
        # self.assertEqual(True, True)


def run_model_unit_tests():
    # absltest.main(__name__)
    testing = TestingMLLM()
    testing.test_mllm_forward_pass()

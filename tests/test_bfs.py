import logging
import unittest

from src.recom_search.model.model_bfs import bfs
from src.recom_search.model.model_output import SearchModelOutput
from src.recom_search.model.setup import process_arg, setup_model


# test of vanilla best first search
class TestBfs(unittest.TestCase):
    def prepare_input(self):
        input_text = "Transformers provides APIs to easily download and train state-of-the-art pretrained models. Using pretrained models can reduce your compute costs, carbon footprint, and save you time from training a model from scratch. The models can be used across different modalities such as: Text: text classification, information extraction, question answering, summarization, translation, and text generation in over 100 languages. Images: image classification, object detection, and segmentation. Our library supports seamless integration between three of the most popular deep learning libraries: PyTorch, TensorFlow and JAX. Train your model in three lines of code in one framework, and load it for inference with another."

        self.input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(
            self.args.device
        )

    def setUp(self) -> None:
        args = process_arg()

        # args.task = 'sum'
        # args.dataset = 'xsum'
        args.hf_model_name = "facebook/bart-large-xsum"
        args.task = "custom"
        args.dataset = "custom_input"
        # args.merge = 'rcb'
        args.merge = "none"

        # args.hf_model_name = 'facebook/bart-large-xsum'
        # args.hf_model_name = 'sshleifer/distilbart-cnn-6-6'
        tokenizer, model, dataset, dec_prefix = setup_model(
            args.task, args.dataset, args.hf_model_name, args.device
        )
        self.model = model
        self.tokenizer = tokenizer
        self.dec_prefix = dec_prefix
        self.param_sim_function = {
            "ngram_suffix": args.ngram_suffix,
            "len_diff": args.len_diff,
            "merge": args.merge,
        }
        self.config_search = {
            "post": args.post,
            "post_ratio": args.post_ratio,  # ratio of model calls left for post finishing
            "dfs_expand": args.dfs_expand,
            "heu": args.use_heu,
        }
        self.args = args
        self.prepare_input()
        logging.info(args)
        return super().setUp()

    def test_bfs(self):
        output = bfs(
            doc_input_ids=self.input_ids,
            model=self.model,
            tokenizer=self.tokenizer,
            dec_prefix=self.dec_prefix,
            avg_score=self.args.avg_score,
            max_len=30,
            k_best=self.args.k_best,
            comp_budget=200,
            config_heu=None,
            config_search=self.config_search,
        )
        mo = SearchModelOutput(ends=output)
        print(mo)


if __name__ == "__main__":
    # begin the unittest.main()
    unittest.main()

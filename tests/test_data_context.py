import pytest
from datasets import Dataset
from languagemodel.context.data_context import Hfdata


class TestHfClass:
    hf_data = Hfdata(
        data=Dataset.from_dict(
            {
                "text": ["Hello world", "Bye bye", "This is a test"]
            }
        ),
        pre_trained_model_name="huggyllama/llama-7b",
    )


class TestHfSubClass(TestHfClass):
    def test_tokenize(self):
        self.hf_data.tokenize(feature_col_name="text",
                              padding=False, truncation=False)
        assert len(self.hf_data.data['input_ids']) == 3
        assert self.hf_data.data["input_ids"][0] == [1, 15043, 3186]
        assert self.hf_data.data["input_ids"][1] == [1, 2648, 29872, 491, 29872]
        assert self.hf_data.data["input_ids"][2] == [1, 910, 338, 263, 1243]

    def test_collate_batch_with_padding(self):
        collated = self.hf_data.collate_batch_with_padding(
            self.hf_data.data.select_columns(["input_ids", "attention_mask"]).to_dict()
        )
        assert len(collated) == 2
        assert len(collated["input_ids"]) == 3
        assert len(collated["input_ids"][0]) == 5
        assert len(collated["input_ids"][1]) == 5
        assert len(collated["input_ids"][2]) == 5

from functools import partial
from typing import (
    Any,
    Optional,
    Union,
)
from datasets import (
    DatasetDict,
    IterableDatasetDict,
    load_dataset,
)
from tokenizers import Tokenizer
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    # DataCollatorForLanguageModeling,
    # DataCollator,
    # DataProcessor,
)


class Hfdata:
    def __init__(
        self,
        data_path: str = "",
        streaming: bool = True,
        batched: bool = True,
        batch_size: int = 1000,
        data: Union[DatasetDict, IterableDatasetDict] = None,
        tokenizer: Tokenizer = None,
        pre_trained_model_name: str = "",
        num_added_special_tokens: int = 0,
    ):
        self.data_path = data_path
        self.streaming = streaming
        self.batched = batched
        self.batch_size = batch_size
        self.data = data
        self.tokenizer = tokenizer
        self.pre_trained_model_name = pre_trained_model_name
        self.num_added_special_tokens = num_added_special_tokens
        # self._initialize()

    def _initialize_tokenizer(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pre_trained_model_name)

    def load(self) -> Union[DatasetDict, IterableDatasetDict]:
        self.data = load_dataset(self.data_path, streaming=self.streaming,
                                 batched=self.batched, batch_size=self.batch_size)

    def collate_batch_with_padding(self, sample: dict[str, Any]):
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # use dynamic padding for each batch
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt",
        )
        return data_collator(sample)

    def _create_labels(self, example: dict[str, Any]) -> dict[str, Any]:
        ...

    def collate_labels(self) -> None:
        ...

    def _tokenize_func(
        self,
        example: dict[str, Any],
        feature_col_name: str,
        padding: bool = False,
        truncation: bool = False,
        return_attention_mask: bool = True,
    ) -> Any:
        if self.tokenizer is None:
            self._initialize_tokenizer()
        return self.tokenizer(
            example[feature_col_name],
            padding=padding,
            truncation=truncation,
            return_attention_mask=return_attention_mask,
        )

    def tokenize(
        self,
        feature_col_name: str,
        padding: bool = False,
        truncation: bool = False,
    ) -> Optional[dict]:
        self.data = self.data.map(
            partial(
                self._tokenize_func,
                feature_col_name=feature_col_name,
                padding=padding,
                truncation=truncation,
            ),
            batched=self.batched
        )

    # def top(self, split: str = "train", token_col_name: str = "input_ids") -> dict[str, list]:
    #     if not self.streaming:
    #         return self.data[split][0][token_col_name]
    #     else:
    #         return next(iter(self.data[split][token_col_name]))

    def sample(self, split: str = "train", token_col_name: str = "input_ids") -> list:
        ...

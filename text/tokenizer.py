from typing import List, Tuple, Union
import torch
from experimaestro import Param
from xpmir.text.huggingface import HFTokenizer
from xpmir.learning.optim import ModuleInitOptions
from xpmir.text.tokenizers import TokenizedTexts, TokenizerOptions
from xpmir.text.huggingface.tokenizers import HFTokenizerAdapter

# Follow the work of ColBERTv2, we insert markers


def _insert_prefix_token(
    tensor: torch.Tensor, prefix_id: Union[int, List], num: int = 1
):
    if isinstance(prefix_id, int):
        prefix_tensor = torch.full(
            (tensor.size(0), num),
            prefix_id,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        assert len(prefix_id) == num
        prefix_tensor = (
            torch.tensor(prefix_id)
            .unsqueeze(0)
            .expand(tensor.size(0), -1)
            .to(dtype=tensor.dtype)
            .to(tensor.device)
        )
    return torch.cat([tensor[:, :1], prefix_tensor, tensor[:, 1:]], dim=1)


# In ColBERTv2, it always pad the query to a fix length whatever the given
# length
class HFTokenizerColBERTQuery(HFTokenizer):
    """The colbert's tokenzier, which pads the query with [MASK]
    to fix length"""

    q_marker: Param[str] = "[unused0]"

    def __initialize__(self, options: ModuleInitOptions):
        super().__initialize__(options)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.q_marker_id = self.tokenizer.convert_tokens_to_ids(self.q_marker)

    def tokenize(
        self,
        texts: List[str] | List[Tuple[str, str]],
        options: TokenizerOptions | None = None,
    ) -> TokenizedTexts:
        options = options or HFTokenizer.DEFAULT_OPTIONS
        max_length = options.max_length
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        else:
            max_length = min(max_length, self.tokenizer.model_max_length)

        r = self.tokenizer(
            list(texts),
            # -1 because we need to pad the marker
            max_length=max_length - 1,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_length=options.return_length,
            return_attention_mask=options.return_mask,
        )

        ids = r["input_ids"]
        ids[ids == self.pad_token_id] = self.mask_token_id
        ids = _insert_prefix_token(ids, self.q_marker_id)
        if options.return_length:
            r["length"] = r["length"] + 1
        if options.return_mask:
            r["attention_mask"] = _insert_prefix_token(r["attention_mask"], 1)
        if r.get("token_type_ids", None) is not None:
            r["token_type_ids"] = _insert_prefix_token(r["token_type_ids"], 0)
        return TokenizedTexts(
            None,
            ids,
            r.get("length", None),
            r.get("attention_mask", None),
            r.get("token_type_ids", None),
        )


class HFTokenizerColBERTDocument(HFTokenizer):
    """The colbert's tokenzier, which pads the query with [MASK]
    to fix length"""

    add_d_tokens: Param[int] = 1
    """The number of additional [D] token prepend, normally initialize to [unused1]
    if the addition number of tokens are provided, the markers should be
    [unused2], [unused3], etc
    """

    def __initialize__(self, options: ModuleInitOptions):
        super().__initialize__(options)
        self.d_markers = [f"[unused{i}]" for i in range(1, self.add_d_tokens + 1)]
        self.d_marker_ids = self.tokenizer.convert_tokens_to_ids(self.d_markers)

    def tokenize(self, texts, options=None):
        options = options or HFTokenizer.DEFAULT_OPTIONS
        max_length = options.max_length
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        else:
            max_length = min(max_length, self.maxtokens())

        r = self.tokenizer(
            list(texts),
            # -1 because we need to pad the marker
            max_length=max_length - self.add_d_tokens,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_length=options.return_length,
            return_attention_mask=options.return_mask,
        )

        ids = r["input_ids"]
        # try to append the d_markers to the mask and the ids
        ids = _insert_prefix_token(ids, self.d_marker_ids, self.add_d_tokens)
        if options.return_length:
            r["length"] = r["length"] + self.add_d_tokens
        if options.return_mask:
            r["attention_mask"] = _insert_prefix_token(
                r["attention_mask"], 1, self.add_d_tokens
            )
        if r.get("token_type_ids", None) is not None:
            r["token_type_ids"] = _insert_prefix_token(
                r["token_type_ids"], 0, self.add_d_tokens
            )

        return TokenizedTexts(
            None,
            ids,
            r.get("length", None),
            r.get("attention_mask", None),
            r.get("token_type_ids", None),
        )


class HFStringTokenizerColBERT(HFTokenizerAdapter):
    """A class which generate different tokenizer instance for query and
    documents"""

    @classmethod
    def from_pretrained_id(cls, hf_id: str, query=True, num_add_tokens=1, **kwargs):
        if query:
            return cls(tokenizer=HFTokenizerColBERTQuery(model_id=hf_id), **kwargs)
        else:
            return cls(
                tokenizer=HFTokenizerColBERTDocument(
                    model_id=hf_id, add_d_tokens=num_add_tokens
                ),
                **kwargs,
            )

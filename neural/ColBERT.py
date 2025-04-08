import json
import os
import string
import torch
from dataclasses import InitVar, dataclass
from typing import NamedTuple, Union, Optional, List
from abc import ABC, abstractmethod

from huggingface_hub import hf_hub_download
from experimaestro import Param, LightweightTask

import torch.nn as nn
from transformers import AutoConfig, PreTrainedModel
from transformers.models.bert import BertModel
from transformers.models.distilbert import DistilBertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.utils import cached_file

from xpmir.text.huggingface import HFModel
from xpmir.text.huggingface.encoders import HFTokensEncoder
from xpmir.utils.utils import easylog, foreach
from xpmir.learning.optim import ModuleInitMode
from xpmir.learning.context import TrainerContext
from xpmir.text import TokenizerOptions
from xpmir.text.encoders import (
    TokenizedTexts,
    TokensRepresentationOutput,
)
from xpmir.neural.dual import DualVectorListener
from xpmir.neural.interaction import InteractionScorer
from xpmir.neural.interaction.common import (
    SimilarityInput,
    SimilarityOutput,
)
from letor.listener import dominate_simple

logger = easylog()

HFConfigName = Union[str, os.PathLike]


@dataclass
class TransformersColBERTOutput(BaseModelOutputWithPoolingAndCrossAttentions):
    """A dataclass include the orginal bert model's last hidden state before
    projection"""

    bert_last_hidden_state: torch.FloatTensor = None


# --- Huggingface Configs
class ColBERTConfig(NamedTuple):
    """ColBERT configuration when loading a pre-trained ColBERT model"""

    dim: int
    query_maxlen: int
    similarity: str
    attend_to_mask_tokens: bool
    data: dict

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: HFConfigName):
        resolved_config_file = cached_file(
            pretrained_model_name_or_path, "artifact.metadata"
        )
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        with open(resolved_config_file, "rt") as fp:
            data = json.load(fp)
            kwargs = {key: data[key] for key in ColBERTConfig._fields if key != "data"}
            config.colbert = ColBERTConfig(**kwargs, data=data)
        return config


class DistilColBERTConfig(NamedTuple):
    """ColBERT configuration when training from scratch, based on DistilColBERT"""

    dim: int

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: HFConfigName):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        config.colbert = DistilColBERTConfig(dim=128)
        return config


# --- Huggingface Models
class ColBERTModel(PreTrainedModel):
    """ColBERT model"""

    DEFAULT_OUTPUT_SIZE = 128

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, config.colbert.dim, bias=False)

    def forward(self, ids, **kwargs):
        output = self.bert(ids, **kwargs)
        bert_last_hidden_state = output.last_hidden_state.clone()
        output.last_hidden_state = self.linear(output.last_hidden_state)
        # store also the bert's last hidden state
        return TransformersColBERTOutput(
            bert_last_hidden_state=bert_last_hidden_state,
            **output,
        )

    @classmethod
    def from_config(cls, config):
        return super(ColBERTModel, cls)._from_config(config)


class DistilColBERTModel(PreTrainedModel):
    """ColBERT model"""

    DEFAULT_OUTPUT_SIZE = 128

    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.linear = nn.Linear(config.hidden_size, config.colbert.dim, bias=False)

    def forward(self, ids, **kwargs):
        output = self.distilbert(ids, **kwargs)
        bert_last_hidden_state = output.last_hidden_state.clone()
        output.last_hidden_state = self.linear(output.last_hidden_state)
        # store also the distilbert's last hidden state
        return TransformersColBERTOutput(
            bert_last_hidden_state=bert_last_hidden_state,
            **output,
        )

    @classmethod
    def from_config(cls, config):
        return super(DistilColBERTModel, cls)._from_config(config)


# --- Xpmir Encoders
class ColBERTEncoder(HFModel):
    model: InitVar[ColBERTModel]
    automodel = ColBERTModel
    autoconfig = ColBERTConfig


class ColBERTEncoderScratch(ColBERTEncoder):
    model: InitVar[ColBERTModel]
    automodel = DistilColBERTModel
    autoconfig = DistilColBERTConfig


# --- Xpmir Encoder with projector on top of it
class ColBERTProjectorAdapter(HFTokensEncoder):
    """The colbert projector, the encoded vector is pseudo-normalized,
    e.g. norm <= 1
    """

    model: Param[ColBERTEncoder]
    """The base colbert encoder"""

    ns_dim: Param[int] = 64
    """The additional dimension"""

    projector_norm: Param[float] = 1e-6
    """The norm for the projector vectors,
    will be buggy if setting to 0
    """

    def __initialize__(self, options):
        super().__initialize__(options)
        hidden_size = self.model.hf_config.hidden_size
        self.projector = nn.Linear(hidden_size, self.ns_dim, bias=False)
        self.zero_projector()
        if isinstance(self.model, ColBERTEncoderScratch):
            # initialize the projector's weight if training from distilbert
            self.model.model.linear.reset_parameters()

    def zero_projector(self):
        self.projector.weight.data = (
            torch.nn.functional.normalize(self.projector.weight.data)
            * self.projector_norm
        )

    def forward(self, tokenized: TokenizedTexts) -> TokensRepresentationOutput:
        tokenized = tokenized.to(self.model.contextual_model.device)
        if isinstance(self.model, ColBERTEncoderScratch):
            # distilbert doesn't have the token type ids
            y: TransformersColBERTOutput = self.model.contextual_model(
                tokenized.ids,
                attention_mask=tokenized.mask.to(self.device),
            )
        else:
            y: TransformersColBERTOutput = self.model.contextual_model(
                tokenized.ids,
                attention_mask=tokenized.mask.to(self.device),
                token_type_ids=tokenized.token_type_ids,
            )
        projected_last_hidden_state = self.projector(y.bert_last_hidden_state)
        ns_last_hidden_state = y.last_hidden_state / torch.sqrt(
            y.last_hidden_state.norm(dim=-1) ** 2
            + projected_last_hidden_state.norm(dim=-1) ** 2
        ).unsqueeze(-1)
        return TokensRepresentationOutput(
            tokenized=tokenized, value=ns_last_hidden_state
        )


# --- Initialization tasks: Modify the params of the model before learning
class ColBERTProjectionInitialization(LightweightTask):
    """Load the projector layer, and then adjust the additional embeddings related
    to the additional tokens"""

    hf_id: Param[str]
    model: Param[ColBERTEncoder]
    num_add_tokens: Param[int] = 1
    """with a number of the additional tokens>1, it will initialize the [unusedi]
    token to the embeddings of the token [unused1]
    the id to copy from is at id [2]

    tried, but doesn't work
    """

    def execute(self):
        self.model.initialize(ModuleInitMode.DEFAULT.to_options())
        cp_path = hf_hub_download(repo_id=self.hf_id, filename="pytorch_model.bin")
        loaded = torch.load(cp_path)
        linear = loaded["linear.weight"]
        self.model.model.linear.weight.data = linear
        if self.num_add_tokens == 1:
            return
        w_embeddings = loaded["bert.embeddings.word_embeddings.weight"]
        w_replacing = w_embeddings[2].unsqueeze(0).expand(self.num_add_tokens, -1)
        w_embeddings[2 : 2 + self.num_add_tokens] = w_replacing
        self.model.model.bert.embeddings.word_embeddings.weight.data = w_embeddings


class ColBERTSecondStageTrainingRandomProjector(LightweightTask):
    """During the DistilBERT second stage training, replace the 0 projector
    norm to a low value, otherwise will be buggy"""

    model: Param[ColBERTProjectorAdapter]

    def execute(self):
        self.model.initialize(ModuleInitMode.DEFAULT.to_options())
        # get the projector
        projector_in = self.model.projector.in_features
        projector_out = self.model.projector.out_features
        new_projector = nn.Linear(projector_in, projector_out, bias=False)
        new_projector.weight.data = (
            torch.nn.functional.normalize(new_projector.weight.data) * 1e-3
        )
        self.model.projector.weight.data = new_projector.weight.data
        logger.info("Replacing the previous 0 projector to a random initialization!")


# --- Model
# --------- Vanilla Version
class ColBERTWithProjector(InteractionScorer):
    """a colbert model with a projector"""

    mask_attend: Param[bool] = True
    """Whether the mask token of the query attend the
    final scoring"""

    mask_punctuation: Param[bool] = True
    """Whether we mask the punctuation for the document tokens during
    matching"""

    num_add_tokens: Param[int] = 1
    """The number of the additional tokens on the document side"""

    def __initialize__(self, options):
        super().__initialize__(options)
        # a trainable parameter to learn to scale the distillation scores
        self.alpha = nn.Parameter(torch.tensor(1.0))
        # a list of the punctuations
        self.skiplist = [
            self.encoder.tokenizer.tokenizer.tokenizer.encode(
                symbol, add_special_tokens=False
            )[0]
            for symbol in string.punctuation
        ]

    def prepare_vanilla_inputs(self, records):
        """return the value and mask after encoding
        with the post processing of punctuation"""
        pad_token_id = self.encoder.tokenizer.tokenizer.tokenizer.pad_token_id
        encoded = self.encoder(
            records,
            options=TokenizerOptions(self.dlen),
        )

        if self.mask_punctuation:
            # not only mask the pad tokens, but also mask all the
            # punctuation tokens
            # The mask will only apply on the scoring stage but during
            # the calculation of the self-attention
            mask = [
                [(x not in self.skiplist) and (x != pad_token_id) for x in d]
                for d in encoded.tokenized.ids.cpu().tolist()
            ]
            mask = (
                torch.tensor(mask)
                .to(dtype=encoded.tokenized.mask.dtype)
                .to(encoded.tokenized.mask.device)
            )
        else:
            mask = encoded.tokenized.mask

        value = encoded.value

        return value, mask

    def encode_documents(self, records):
        value, mask = self.prepare_vanilla_inputs(records)
        return self.similarity.preprocess(
            SimilarityInput(
                value=value,
                mask=mask,
            )
        )

    def merge(self, objects: List[SimilarityInput]):
        # Used to merge the query terms
        # As in colbert, all the queries are append to a fix length
        # using a simple concatenate is OK.
        mask = torch.cat([object.mask for object in objects], dim=0)
        value = torch.cat([object.value for object in objects], dim=0)

        return SimilarityInput(
            value=value,
            mask=mask,
        )

    def compute_scores(
        self,
        queries: SimilarityInput,
        documents: SimilarityInput,
        value: SimilarityOutput,
        info: Optional[TrainerContext] = None,
    ):
        # Similarity matrix B x Lq x Ld or Bq x Lq x Bd x Ld
        # In vanilla colbertv2 they don't mask the queries
        s = value.similarity.masked_fill(
            value.d_view(documents.mask).logical_not(), float("-inf")
        )
        if not self.mask_attend:
            s = s.masked_fill(value.q_view(queries.mask).logical_not(), 0)

        # call the hooks -- Loggers or Regularizations
        # Apply the dual vector hook
        if info is not None:
            foreach(
                info.hooks(DualVectorListener),
                lambda hook: hook(info, queries, documents),
            )

        return nn.functional.relu(s).max(-1).values.sum(1).flatten()


# --------- Abstract Class For Pruning
class ColBERTWithProjectorPruning(ColBERTWithProjector, ABC):
    @abstractmethod
    def pruning_tokens(self, original_mask, original_value) -> torch.Tensor:
        """Define the pruning strategy"""
        ...

    def encode_documents(self, records):
        value, mask = self.prepare_vanilla_inputs(records)
        # Apply the pruning mask
        mask = self.pruning_tokens(mask, value)
        return self.similarity.preprocess(
            SimilarityInput(
                value=value,
                mask=mask,
            )
        )


# --------- Pruning based on Norms
class ColBERTWithProjectorNormPruning(ColBERTWithProjectorPruning):
    """a colbert model with a projector and pruning based on the norm"""

    norm_threshold: Param[float] = 0.1

    def pruning_tokens(self, original_mask, original_value):
        norm_mask = original_value.norm(dim=-1) >= self.norm_threshold
        mask = torch.logical_and(norm_mask, original_mask)
        return mask


# --------- Pruning based on Linear Programming (LP)
class ColBERTWithLPPruning(ColBERTWithProjectorPruning):

    lp_threshold: Param[float] = 0.6

    @torch.no_grad()
    def pruning_tokens(self, original_mask, original_value):
        _, S, Vh = torch.linalg.svd(
            # shape [bs, dim, length]
            (original_value * original_mask.unsqueeze(-1)).transpose(1, 2),
            full_matrices=False,
        )
        normalized_S = S / S.sum(-1).unsqueeze(-1)
        cumS = torch.cumsum(normalized_S, dim=-1)
        # reduced dimensions
        k = (cumS <= self.lp_threshold).sum(-1)
        k = k + 1
        output_mask = original_mask.clone().detach().cpu()
        Vh = Vh.cpu().detach()
        for indice, (D_prime, mask, reduced_dim) in enumerate(zip(Vh, output_mask, k)):
            nonzero_indice = torch.where(mask != 0)
            D_prime = D_prime[:reduced_dim]
            D_prime = D_prime[:, nonzero_indice[0]]
            max_values, max_indice = torch.max(D_prime, dim=-1)
            min_values, min_indice = torch.min(D_prime, dim=-1)
            pre_filter = max_indice[torch.where(max_values > 0)]
            pre_filter = torch.cat(
                (pre_filter, min_indice[torch.where(min_values < 0)])
            )
            pre_filter = pre_filter.unique()
            pruned_ids = []
            for i in range(D_prime.shape[1]):
                if i in pre_filter:
                    continue
                res = dominate_simple(D_prime, i, 0)
                if res:
                    pruned_ids.append(i)
            pruned_tokens_indices = nonzero_indice[0][pruned_ids]
            mask[pruned_tokens_indices] = 0
            output_mask[indice] = mask
        return output_mask.to(dtype=original_mask.dtype).to(original_mask.device)

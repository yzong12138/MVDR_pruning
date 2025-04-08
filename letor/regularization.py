from experimaestro import Param
import torch

from xpmir.learning.context import TrainerContext, Loss
from xpmir.learning.metrics import ScalarMetric
from xpmir.neural.dual import DualVectorListener
from xpmir.neural.interaction.common import SimilarityInput

# --- the Norm regularization


class ColBERTProjectorDocumentNormRegularization(DualVectorListener):
    """The regularization hook to decrease the norm of the last hidden state
    of the document"""

    coeff: Param[float]
    """The coeff for this regularization"""

    norm_type: Param[float] = 2
    """The norm type shows whether we need to use the l1 norm(1) or the
    l2 norm(2, fro)"""

    def __call__(
        self,
        context: TrainerContext,
        queries: SimilarityInput,
        documents: SimilarityInput,
    ):
        d_values = documents.value
        d_loss = torch.sum(
            torch.norm(
                d_values * documents.mask.unsqueeze(-1), p=self.norm_type, dim=-1
            )
        ) / torch.sum(documents.mask)
        context.add_loss(
            Loss(f"Document-Projector-Residual-L{self.norm_type}", d_loss, self.coeff)
        )


class ColBERTProjectorQueryNormRegularization(DualVectorListener):
    """The regularization hook to decrease the norm of the last hidden state
    of the query, and try to log out the variance"""

    coeff: Param[float]
    """The coeff for this regularization"""

    mask_attend: Param[bool] = True
    """
    When this value is true, means the mask tokens in the query will
    also attend the final score, which means that in this regularization
    we also need to optimize the mask tokens of query

    When this value is false, we ignore the query on the query side.
    """

    norm_type: Param[float] = 2
    """The norm type shows whether we need to use the l1 norm(1) or the
    l2 norm(2, fro)"""

    def __call__(
        self,
        context: TrainerContext,
        queries: SimilarityInput,
        documents: SimilarityInput,
    ):
        if self.mask_attend:
            q_loss = torch.mean(torch.norm(queries.value, p=self.norm_type, dim=-1))
            q_var = torch.mean(torch.var(queries.value, dim=-1))
        else:
            q_loss = torch.sum(
                torch.norm(
                    (queries.value) * queries.mask.unsqueeze(-1),
                    p=self.norm_type,
                    dim=-1,
                )
            ) / torch.sum(queries.mask)
            # todo: need to mask the masked token for the variance
            q_var = torch.mean(torch.var(queries.value, dim=-1))

        context.add_loss(
            Loss(f"Query-Projector-Residual-L{self.norm_type}", q_loss, self.coeff)
        )
        # log the variance of the variance on the query side
        context.add_metric(
            ScalarMetric(
                "Query-Projector-Variance",
                q_var,
                1,
            )
        )


# --- the Nuclear Norm Regularization
class ColBERTSVDDiagonalMinmization(DualVectorListener):
    """The ColBERT Singular Value Minimization based on the SVD decomposition,
    a.k.a. the nuclear norm

    Theoritically, this regularization tries to put the
    directions into the same direction as the other vectors.
    """

    coeff: Param[float]
    """The coeff for this regularization"""

    def __call__(
        self,
        context: TrainerContext,
        queries: SimilarityInput,
        documents: SimilarityInput,
    ):
        value = documents.value
        mask = documents.mask
        device = value.device
        encoder_dim = value.shape[-1]
        value_masked = value * mask.unsqueeze(-1)

        s = torch.linalg.matrix_norm(value_masked.transpose(1, 2), ord="nuc")
        s_length = torch.min(
            mask.sum(-1), torch.tensor([encoder_dim]).unsqueeze(-1).to(device)
        )
        loss = (s / s_length).mean()

        context.add_loss(
            Loss(
                "Nuclear-Norm-Regu",
                loss,
                self.coeff,
            )
        )


# --- The Document Tokens Similarity Matrix Regularization
class ColBERTDocVecSimRegularization(DualVectorListener):
    """The ColBERT minimization regularization based on the document
    vectors' similarity.
    It tries to do the similar things as the svd based loss and it is more
    heuristic but it is faster and could be more stable.
    """

    coeff: Param[float]
    """The coeff for this regularization"""

    eps: Param[float] = 1e-2
    """The eps to stabilize the gradients"""

    def __call__(
        self,
        context: TrainerContext,
        queries: SimilarityInput,
        documents: SimilarityInput,
    ):
        value = documents.value
        mask = documents.mask
        device = value.device
        norm = value.norm(dim=-1)
        norm_multiplier = (norm - 1) / (norm + self.eps)
        doc_inner_sim = torch.bmm(value, value.transpose(1, 2))
        diag_mask = torch.eye(doc_inner_sim.shape[-1]).to(device)
        sim_part = (
            doc_inner_sim * mask.unsqueeze(1) * diag_mask.logical_not().unsqueeze(0)
        ).sum(-1) * mask
        loss = ((norm_multiplier * sim_part).sum(-1) / (mask.sum(-1) ** 2)).mean()
        context.add_loss(
            Loss(
                "Document-Vector-Similarity-Loss",
                loss,
                self.coeff,
            )
        )

import torch
from typing import List
from tqdm import tqdm
import json
from scipy.optimize import linprog
from experimaestro import Param, Meta
from datamaestro_text.data.ir import DocumentStore
from xpmir.evaluation import evaluate
from xpmir.learning.context import TrainState
from xpmir.learning.learner import LearnerListener, LearnerListenerStatus, Learner
from xpmir.letor.learner import ValidationListener, ValidationModuleLoader
from xpmir.utils.utils import easylog, foreach

logger = easylog()


def dominate_simple(D: torch.Tensor, k: int, eps: float = 0):
    """A simple method to test if a document token is dominated by
    considering the initial dominance condition
    D is the input matrix of shape [encoder_dim, length], the mask tokens
    are already removed from the matrix k the kth column of the model.
    eps is a value <=0, as the lower bound for the variable
    """
    _, d_length = D.shape
    z = D[:, k]

    # Objective function (minimize 0 since there's no explicit objective)
    c = torch.zeros(d_length - 1)

    # Constraints y >= 0
    bounds = [(eps, None)] * (d_length - 1)  # y >= 0 for each component of y

    # Equality constraints:
    # (D-z1^t)y = z
    # We need to form the constraint matrix A_eq and vector b_eq
    A_eq = D - z.unsqueeze(-1)
    A_eq = torch.cat((A_eq[:, :k], A_eq[:, k + 1 :]), dim=1)
    b_eq = z

    # Now solve the linear program
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    # Output the result
    return result.success


class DominanceRateLoggerListener(LearnerListener):

    documents: Param[DocumentStore]
    """The document set to sample the documents to test"""

    logger_interval: Param[int] = 10
    """Epochs between each logging"""

    samples: Meta[int] = 256
    """The counts of the documents we need to use for computing"""

    use_ratio: Param[bool] = True
    """if true, use the cum sigma, else use the absolute sigma"""

    dim_redu_eps: Param[float] = 0.5
    """The eps for the dimension reduction"""

    cum_sigma_redu_eps: Param[float] = 0.7
    """The eps for the cumulate normalized singular values to
    reduce the dimension
    """

    current_metric: float
    """A variable to store the LP pruning ratio during the validation"""

    @torch.no_grad()
    def __call__(self, state: TrainState):
        if state.epoch % self.logger_interval == 0:
            logger.info("Start to do the LP dominance test")
            doc_count = self.documents.documentcount
            int_ids = torch.randperm(doc_count)[: self.samples].tolist()
            # assume that in MSMARCO, the ext ids are the stringfied
            # integers, could be buggy if the given corpus are the others
            test_doc = self.documents.documents_ext([str(int_id) for int_id in int_ids])
            state.model.eval()
            res = state.model.encode_documents(test_doc)

            _, S, Vh = torch.linalg.svd(
                # shape [bs, dim, length]
                (res.value * res.mask.unsqueeze(-1)).transpose(1, 2),
                full_matrices=False,
            )
            if self.use_ratio:
                normalized_S = S / S.sum(-1).unsqueeze(-1)
                cumS = torch.cumsum(normalized_S, dim=-1)
                k = (cumS <= self.cum_sigma_redu_eps).sum(-1)
                k = k + 1
            else:
                k = (S > self.dim_redu_eps).sum(-1)
            domi_count = 0
            for (D_prime, mask, reduced_dim) in tqdm(
                zip(Vh, res.mask, k), total=self.samples
            ):
                nonzero_indice = torch.where(mask != 0)
                D_prime = D_prime[:reduced_dim]
                D_prime = D_prime[:, nonzero_indice[0]].cpu().detach()
                for i in range(D_prime.shape[1]):
                    domi_count += dominate_simple(D_prime, i, 0)
            if self.use_ratio:
                self.context.writer.add_scalar(
                    f"LP_dominance_cum_ratio_{self.cum_sigma_redu_eps}_new",
                    float(domi_count / res.mask.sum()),
                    state.step,
                )
            else:
                self.context.writer.add_scalar(
                    "LP_dominance_ratio",
                    float(domi_count / res.mask.sum()),
                    state.step,
                )
            self.set_metrics(float(domi_count / res.mask.sum()))
            logger.info("LP dominance test finish")

        return LearnerListenerStatus.NO_DECISION

    def set_metrics(self, metric):
        self.current_metric = metric

    def get_metric(self):
        return self.current_metric


class DominaceValidationInteractionListener(ValidationListener):
    """We store the checkpoint with the best Dominance value and the best
    evaluation score on IR, equilibrate with weighted harmonic value

    The dominance is calculate through the dominace listener, and the
    IR metric is calculate with the current listener
    """

    dominance_listener: Param[DominanceRateLoggerListener]
    """The dominance listener which log out the pruning ratio"""

    f_values: Param[List[float]] = [0.5, 1, 2, 4, float("+inf")]
    """The f_values we calculate, the smaller the value,
    the more it converge to the ir_metric"""

    f_names: Param[List[str]] = ["F0.5", "F1", "F2", "F4", "Finf"]

    def f_scores(self, ir_metric, lp_metric):
        # also add the +inf at the end. ==> pure lp_metric
        f_tensors = torch.tensor(self.f_values[:-1])
        res = ((1 + f_tensors**2) * ir_metric * lp_metric) / (
            (f_tensors**2 * ir_metric) + lp_metric
        )
        res = res.tolist()
        res.append(lp_metric)
        return res

    def initialize(self, learner, context):
        super().initialize(learner, context)
        self.dominance_listener.initialize(learner, context)

    def monitored(self):
        # get all the names of the storing metrics
        base_ir_metrics = [key for key, store in self.metrics.items() if store]

        F_keys = [
            f"{ir_metric}_LP_{f_name}"
            for ir_metric in base_ir_metrics
            for f_name in self.f_names
        ]
        return base_ir_metrics + F_keys

    def init_task(self, learner: "Learner", dep):
        all_stored_key_names = self.monitored()
        return {
            key: dep(
                ValidationModuleLoader(
                    value=learner.model,
                    listener=self,
                    key=key,
                    path=self.bestpath / key / TrainState.MODEL_PATH,
                )
            )
            for key in all_stored_key_names
        }

    def __call__(self, state: TrainState):
        foreach(
            self.hooks,
            lambda hook: hook.before(self.context),
        )

        if state.epoch % self.validation_interval == 0:
            # Compute validation metrics
            self.dominance_listener(state)
            lp_ratio = self.dominance_listener.get_metric()
            means, details = evaluate(
                self.retriever, self.dataset, list(self.metrics.keys()), True
            )
            # should have only one which is keep in this scenario on the pure ir-metrics
            for metric, keep in self.metrics.items():
                value = means[metric]
                # add the scaler
                self.context.writer.add_scalar(
                    f"{self.id}/{metric}/mean", value, state.step
                )

                # build a list of the values and the names
                metric_values = [value]
                metric_names = [metric]
                if keep:
                    f_metrics = self.f_scores(value, lp_ratio)
                    for f_metric, f_name in zip(f_metrics, self.f_names):
                        self.context.writer.add_scalar(
                            f"{self.id}/{metric}_LP_{f_name}/mean", f_metric, state.step
                        )
                        metric_values.append(f_metric)
                        metric_names.append(f"{metric}_LP_{f_name}")

                if self.should_update_validation(state):
                    for metric_name, metric_value in zip(metric_names, metric_values):
                        topstate = self.top.get(metric_name, None)
                        if topstate is None or metric_value > topstate["value"]:
                            # Save the new top JSON
                            self.top[metric_name] = {
                                "value": metric_value,
                                "epoch": self.context.epoch,
                            }

                            if keep:
                                logger.info(
                                    f"Saving the checkpoint {state.epoch}"
                                    f" for metric {metric_name}"
                                )
                                self.context.copy(self.bestpath / metric_name)

            # Update information
            with self.info.open("wt") as fp:
                json.dump(self.top, fp)

        foreach(
            self.hooks,
            lambda hook: hook.after(self.context),
        )

        # Don't apply the early stop for the moment.
        return LearnerListenerStatus.NO_DECISION

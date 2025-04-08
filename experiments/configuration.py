from attrs import Factory, field
from typing import List
from functools import cached_property

from xpmir.papers import configuration
from experimaestro.experiments.configuration import ConfigurationBase
from xpmir.papers.helpers import LauncherSpecification
from functools import cached_property as attrs_cached_property
from xpmir.letor import Random
from xpmir.learning.devices import Device, CudaDevice
from xpmir.papers.helpers.samplers import ValidationSample
from xpmir.papers.helpers.optim import TransformerOptimization
from xpmir.learning.optim import (
    get_optimizers,
    ParameterOptimizer,
    RegexParameterFilter,
)


@configuration()
class Validation(ValidationSample):
    """Define the details of the validation setup"""

    validation_interval: int = field(default=32)
    """"The interval of the validaion,
    the exact number of steps = validation_interval * steps_per_epoch"""
    validation_top_k: int = 1000
    """The top_k to rerank during the validation"""


@configuration()
class Indexation(LauncherSpecification):
    requirements: str = "duration=1 days & cpu(cores=8)"
    """experimaestro launcher for indexation, check:
    https://experimaestro-python.readthedocs.io/en/latest/launchers/
    """
    max_docs: int = 0
    """Maximum number of indexed documents: should be 0 when not debugging"""


@configuration()
class ProjectorNormReguConfig:
    """The regularization hyper-parameters to minimize norm after
    the projector
    """

    coeff_d: float = 0.01
    """The coefficient for the regularization to minimize norm
    after the projector.
    """
    coeff_q: float = 0.01
    """The coefficient for the regularization to maximize or minimize norm,
    for the query side only
    """
    q_type: float = 1
    """
    On the QUERY SIDE
    The norm type shows whether we need to use the l1 norm(1) or the
    l2 norm(2)"""
    d_type: float = 1
    """
    On the DOCUMENT SIDE
    The norm type shows whether we need to use the l1 norm(1) or the
    l2 norm(2)"""


@configuration()
class NucNormReguConfig:
    """The regulariztion by minimizing the nuclear norm"""

    coeff: float = 0.2
    """The coefficient of the regularization by minimizing the value of the
    diagonal of the singular value matrix, a.k.a. the nuclear norm"""


@configuration()
class DocSimReguConfig:
    """The regularization by minmizing the distance between the document
    tokens"""

    coeff: float = 0.8
    """The coefficient for the regularization by minmizing the distance between
    the document tokens"""


@configuration()
class RegularizationCoeffConfiguration:
    proj_norm_regu: ProjectorNormReguConfig = Factory(ProjectorNormReguConfig)
    """Corresponding to the regularization to decrease the token norm"""

    nuc_norm_regu: NucNormReguConfig = Factory(NucNormReguConfig)
    """Corresponding to the regularization to decrease the nuclear norm"""

    doc_sim_regu: DocSimReguConfig = Factory(DocSimReguConfig)
    """Corresponding to the regularization to decrease the document similarity"""


@configuration()
class ColBERTTokenizeOption:
    qlen: int = 32
    dlen: int = 180
    dlen_OoD: int = 300


@configuration()
class ColBERTTrain:
    distil_data_path: str = "path/to/distil_data"
    """The path of the distillation training data
    """
    need_distil: bool = True
    """Whether train the model with KL div loss
    """
    need_ibn: bool = True
    """Whether train the model with In batch negative loss
    """
    nway: int = 16
    """The number of distillation samples for one query
    """
    reg: RegularizationCoeffConfiguration = Factory(RegularizationCoeffConfiguration)
    """The configuration about the regularization"""


@configuration()
class ColbertProjectorConfiguration:
    ns_dim_doc: int = 32
    """the additional dimension for the pseudo normalize"""

    eps: float = 1e-3
    """The eps for the initial coeff norm
    if = 0 also means we want to freeze the projector"""


@configuration()
class ColBERT:

    hf_id: str = "colbert-ir/colbertv2.0"
    """Identifier for the base model"""

    from_scratch: bool = False
    """If true means we train from the distilbert-based model, without any colbert
    pre-training
    """

    mask_attend: bool = True
    """Whether the mask query token attend the final score"""

    doc_additional_tokens: int = 1
    """the number of the additional [D] tokens we prepend to the
    document tokenization stage"""

    tokenize_option: ColBERTTokenizeOption = Factory(ColBERTTokenizeOption)
    """The tokenization option for ColBERT"""

    colbert_train: ColBERTTrain = Factory(ColBERTTrain)
    """The training related hyperparameters"""

    colbert_projector: ColbertProjectorConfiguration = Factory(
        ColbertProjectorConfiguration
    )
    """The hyperparmeters related to the projector"""


@configuration()
class ColBERTOptimization(TransformerOptimization):
    """This optimization propose a optimizer with a different lr for the
    alpha"""

    alpha_lr: float = 5.0e-3
    """The learning rate for the alpha"""

    alpha_param_name: List[str] = ["alpha"]

    @cached_property
    def optimizer(self):
        if not self.re_no_l2_regularization:
            return get_optimizers(
                [
                    ParameterOptimizer(
                        scheduler=self.scheduler_instance,
                        optimizer=self.get_optimizer(False, self.alpha_lr),
                        filter=RegexParameterFilter(includes=self.alpha_param_name),
                    ),
                    ParameterOptimizer(
                        scheduler=self.scheduler_instance,
                        optimizer=self.get_optimizer(True, self.lr),
                    ),
                ]
            )

        return get_optimizers(
            [
                ParameterOptimizer(
                    scheduler=self.scheduler_instance,
                    optimizer=self.get_optimizer(False, self.alpha_lr),
                    filter=RegexParameterFilter(includes=self.alpha_param_name),
                ),
                ParameterOptimizer(
                    scheduler=self.scheduler_instance,
                    optimizer=self.get_optimizer(False, self.lr),
                    filter=RegexParameterFilter(includes=self.re_no_l2_regularization),
                ),
                ParameterOptimizer(
                    scheduler=self.scheduler_instance,
                    optimizer=self.get_optimizer(True, self.lr),
                ),
            ]
        )


@configuration()
class ColBERTLearner:
    optimization: ColBERTOptimization = Factory(ColBERTOptimization)
    """The parameters for the optimization"""
    requirements: str = "duration=5 days & cuda(mem=40G)"
    """experimaestro launcher for learning, check:
    https://experimaestro-python.readthedocs.io/en/latest/launchers/
    """


@configuration()
class PruningConfig:

    evaluation_norm_threshold: List[float] = [0.1]
    """the threshold for norm"""

    evaluation_LP_ratio_threshold: List[float] = [0.7]
    """the threshold for LP"""


@configuration()
class EvaluationRetrieval:
    k: int = 100
    """Top k for reranking during the retrieval"""
    batch_size: int = 256
    """The batchsize we use for model inference"""
    requirements: str = "duration=5 days & cuda(mem=24G)"
    """experimaestro launcher for evaluation, check:
    https://experimaestro-python.readthedocs.io/en/latest/launchers/
    """


@configuration()
class OoDSet:
    name: List[str] = []
    """the name of the dataset"""
    documents: List[str] = []
    """the documents collection id of the dataset in ir-datasets, check:
    https://ir-datasets.com/
    """
    test: List[str] = []
    """the test set id of the dataset in ir-datasets, check:
    https://ir-datasets.com/
    """

    def check_dataset(self):
        # basic of the coherence of the dataset
        assert len(self.name) == len(self.documents)
        assert len(self.name) == len(self.test)


@configuration()
class Evaluation:
    evaluate_in_domain: bool = True
    """If true, evaluate on msmarco, else on choosen beir + lotte"""

    using_best_validate: bool = True
    """If this is true we use the best validated checkpoint, else
    we use the last checkpoint"""

    evaluate_metric_names: List[str] = []
    """The name of the validation metrics we use for retrieval, if empty means
    use all cps
    only useful when using_best_validate = True
    """

    pruning: PruningConfig = Factory(PruningConfig)
    """The hyperparameters related to the pruning strategy"""

    ood_set: OoDSet = Factory(OoDSet)
    """The Out-of-Domain evaluation set"""

    retrieval: EvaluationRetrieval = Factory(EvaluationRetrieval)
    """The retrieval hyperparameters for evaluation"""


@configuration()
class Preprocessing:
    requirements: str = "duration=12h & cpu(cores=12)"
    """experimaestro launcher for data preprocessing, check:
    https://experimaestro-python.readthedocs.io/en/latest/launchers/
    """


@configuration()
class ColBERTConfiguration(ConfigurationBase):
    # misc
    gpu: bool = True
    """Use GPU for computation"""

    seed: int = 0
    """The seed used for experiments"""

    @attrs_cached_property
    def random(self):
        return Random(seed=self.seed)

    @attrs_cached_property
    def device(self) -> Device:
        return CudaDevice() if self.gpu else Device()

    validation: Validation = Factory(Validation)
    indexation: Indexation = Factory(Indexation)
    colbert: ColBERT = Factory(ColBERT)
    learning: ColBERTLearner = Factory(ColBERTLearner)
    preprocessing: Preprocessing = Factory(Preprocessing)
    evaluation: Evaluation = Factory(Evaluation)

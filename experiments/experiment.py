import logging
from pathlib import Path

# from functools import partial
from experimaestro.launcherfinder import find_launcher
from datamaestro import prepare_dataset

from xpmir.learning.learner import Learner
from xpmir.learning.batchers import Batcher, PowerAdaptativeBatcher
from xpmir.learning.optim import GradientLogHook

from xpmir.models import AutoModel
from xpmir.learning.hooks import LayerFreezer
from xpmir.learning.parameters import RegexParametersIterator

from xpmir.papers.helpers.samplers import (
    prepare_collection,
    msmarco_v1_validation_dataset,
)

from xpmir.letor.trainers.batchwise import SoftmaxCrossEntropy
from xpmir.datasets.adapters import MemoryTopicStore
from xpmir.text.encoders import TokenizedTextEncoder
from xpmir.text.adapters import TopicTextConverter
from xpmir.index.sparse import SparseRetriever, SparseRetrieverIndexBuilder

from xpmir.neural.interaction.common import DotProductSimilarity
from xpmir.experiments.ir import ir_experiment, IRExperimentHelper
from xpmir.utils.functools import cache
from dataset.samplers import (
    DistillationInBatchNegativesSampler,
    DistillationListwiseSampler,
)
from dataset.distillation import ListwiseHydrator, JSONBasedBatchDistillationSamples
from letor.trainer import (
    DistillationInBatchNegativeTrainer,
    DistillationBatchwiseKLLoss,
)
from letor.regularization import (
    ColBERTProjectorDocumentNormRegularization,
    ColBERTSVDDiagonalMinmization,
    ColBERTProjectorQueryNormRegularization,
    ColBERTDocVecSimRegularization,
)
from text.tokenizer import HFStringTokenizerColBERT
from neural.ColBERT import (
    ColBERTEncoder,
    ColBERTEncoderScratch,
    ColBERTProjectionInitialization,
    ColBERTProjectorAdapter,
    ColBERTWithProjector,
    ColBERTWithProjectorNormPruning,
    ColBERTWithLPPruning,
)
from evaluation.evaluation import (
    msmarco_v1_tests,
    OoD_tests,
)
from experiments.configuration import ColBERTConfiguration
from letor.listener import (
    DominanceRateLoggerListener,
    DominaceValidationInteractionListener,
)

logging.basicConfig(level=logging.INFO)


@cache
def msmarco_colbert_distillation_samples(path, nway) -> DistillationListwiseSampler:
    """Distillation samples from ColBERTv2 training."""
    # Access to topic text
    train_topics = prepare_dataset("irds.msmarco-passage.train.queries")

    # Combine the training triplets with the document and queries texts
    distillation_samples = ListwiseHydrator(
        samples=JSONBasedBatchDistillationSamples(
            id="colbertv2.distillation",
            path=Path(path),
            nway_num=nway,
        ),
        documentstore=prepare_collection("irds.msmarco-passage.documents"),
        querystore=MemoryTopicStore(topics=train_topics),
    )

    # Generate a sampler from the samples
    return DistillationListwiseSampler(samples=distillation_samples)


@ir_experiment()
def run(
    helper: IRExperimentHelper,
    cfg: ColBERTConfiguration,
):
    # define the launchers
    launcher_learner = find_launcher(cfg.learning.requirements)
    launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)
    launcher_evaluate = find_launcher(cfg.evaluation.retrieval.requirements)
    launcher_index = find_launcher(cfg.indexation.requirements)

    # misc
    device = cfg.device
    random = cfg.random

    # prepare the training set
    documents = prepare_collection("irds.msmarco-passage.documents")
    ds_val_rerank = msmarco_v1_validation_dataset(
        cfg.validation, launcher=launcher_preprocessing
    )

    # validation retriever based on splade
    spladev2, splade_init_tasks = AutoModel.load_from_hf_hub("xpmir/SPLADE_DistilMSE")

    # build the splade index for retrieving
    sparse_index = SparseRetrieverIndexBuilder(
        batch_size=512,
        batcher=PowerAdaptativeBatcher(),
        encoder=spladev2.encoder,
        device=device,
        documents=documents,
        ordered_index=False,
        max_docs=cfg.indexation.max_docs,
    ).submit(launcher=launcher_index, init_tasks=splade_init_tasks)

    splade_retriever = SparseRetriever(
        index=sparse_index,
        topk=cfg.validation.validation_top_k,
        batchsize=1,
        encoder=spladev2._query_encoder,
        in_memory=True,
    )

    # ----- Building the model
    # --- tokenizers
    converter = TopicTextConverter()
    document_tokenizer = HFStringTokenizerColBERT.from_pretrained_id(
        cfg.colbert.hf_id,
        query=False,
        num_add_tokens=cfg.colbert.doc_additional_tokens,
        converter=converter,
    )
    query_tokenizer = HFStringTokenizerColBERT.from_pretrained_id(
        cfg.colbert.hf_id,
        query=True,
        converter=converter,
    )
    # --- base encoders
    if cfg.colbert.from_scratch:
        base_colbert_encoder = ColBERTEncoderScratch.from_pretrained_id(
            cfg.colbert.hf_id
        )
    else:
        base_colbert_encoder = ColBERTEncoder.from_pretrained_id(cfg.colbert.hf_id)

    # Document and query side share the same encoder, but different tokenizer
    document_token_encoder: ColBERTProjectorAdapter = ColBERTProjectorAdapter(
        model=base_colbert_encoder,
        ns_dim=cfg.colbert.colbert_projector.ns_dim_doc,
        projector_norm=cfg.colbert.colbert_projector.eps,
    )
    colbert_document_encoder: TokenizedTextEncoder = TokenizedTextEncoder(
        tokenizer=document_tokenizer,
        encoder=document_token_encoder,
    )
    colbert_query_encoder: TokenizedTextEncoder = TokenizedTextEncoder(
        tokenizer=query_tokenizer,
        encoder=document_token_encoder,
    )
    # put them together
    colbert: ColBERTWithProjector = ColBERTWithProjector(
        encoder=colbert_document_encoder,
        query_encoder=colbert_query_encoder,
        similarity=DotProductSimilarity(),
        qlen=cfg.colbert.tokenize_option.qlen,
        dlen=cfg.colbert.tokenize_option.dlen,
        mask_attend=cfg.colbert.mask_attend,
        mask_punctuation=True,
        num_add_tokens=cfg.colbert.doc_additional_tokens,
    )

    batcher = (
        Batcher() if cfg.colbert.colbert_train.need_ibn else PowerAdaptativeBatcher()
    )

    # ----- Define some other learning components
    # prepare the regularizations
    training_hooks = []
    if cfg.colbert.colbert_train.reg.nuc_norm_regu.coeff > 0:
        training_hooks.append(
            ColBERTSVDDiagonalMinmization(
                coeff=cfg.colbert.colbert_train.reg.nuc_norm_regu.coeff
            )
        )
    if cfg.colbert.colbert_train.reg.doc_sim_regu.coeff > 0:
        training_hooks.append(
            ColBERTDocVecSimRegularization(
                coeff=cfg.colbert.colbert_train.reg.doc_sim_regu.coeff
            )
        )
    if cfg.colbert.colbert_train.reg.proj_norm_regu.coeff_d > 0:
        training_hooks.append(
            ColBERTProjectorDocumentNormRegularization(
                coeff=cfg.colbert.colbert_train.reg.proj_norm_regu.coeff_d,
                norm_type=cfg.colbert.colbert_train.reg.proj_norm_regu.d_type,
            )
        )
        training_hooks.append(
            ColBERTProjectorQueryNormRegularization(
                coeff=cfg.colbert.colbert_train.reg.proj_norm_regu.coeff_q,
                norm_type=cfg.colbert.colbert_train.reg.proj_norm_regu.q_type,
            )
        )

    assert len(training_hooks) > 0, "no regularization provided!"

    # define the trainer
    colbert_trainer = DistillationInBatchNegativeTrainer(
        sampler=DistillationInBatchNegativesSampler(
            sampler=msmarco_colbert_distillation_samples(
                path=cfg.colbert.colbert_train.distil_data_path,
                nway=cfg.colbert.colbert_train.nway,
            )
        ),
        batcher=batcher,
        batch_size=cfg.learning.optimization.batch_size,
        lossfn=SoftmaxCrossEntropy(),
        lossfn_distillation=DistillationBatchwiseKLLoss(),
        hooks=training_hooks,
        need_ibn=cfg.colbert.colbert_train.need_ibn,
        need_distil=cfg.colbert.colbert_train.need_distil,
    )

    # define the validation listener
    listeners = []
    lp_dominance_logger_listener = DominanceRateLoggerListener(
        id="lp_dominance_logger",
        documents=documents,
        logger_interval=cfg.validation.validation_interval,
        samples=1024,
        use_ratio=True,
        cum_sigma_redu_eps=0.7,
    )
    validation_rerank_splade = DominaceValidationInteractionListener(
        id="colbert_reranker_validation_splade",
        dataset=ds_val_rerank,
        retriever=colbert.getRetriever(
            retriever=splade_retriever,
            batch_size=cfg.evaluation.retrieval.batch_size,
            batcher=PowerAdaptativeBatcher(),
            device=device,
        ),
        validation_interval=cfg.validation.validation_interval,
        metrics={"RR@10": True, "AP": False, "nDCG@10": False},
        dominance_listener=lp_dominance_logger_listener,
    )
    listeners.append(validation_rerank_splade)

    # define a learning hook, initialize with a gradient logger
    hooks = [
        GradientLogHook(name="gradient_norm"),
    ]

    # if we want to freeze the projector,
    # e.g. when the first stage learning of distilbert
    if cfg.colbert.colbert_projector.eps == 0:
        hooks.append(
            LayerFreezer(  # freezing the transformers and projection
                selector=RegexParametersIterator(
                    regex=r"""projector\.""",
                    model=colbert.encoder.encoder,
                ),
            ),
        )

    # ----- Learning
    learner = Learner(
        # Misc settings
        device=device,
        random=random,
        # How to train the model
        trainer=colbert_trainer,
        # The model to train (splade contains all the parameters)
        model=colbert,
        # use mixed precision
        use_fp16=True,
        # Optimization settings
        steps_per_epoch=cfg.learning.optimization.steps_per_epoch,
        optimizers=cfg.learning.optimization.optimizer,
        max_epochs=cfg.learning.optimization.max_epochs,
        # The listeners (here, for validation)
        # listeners=[validation_rerank_splade, validation_first_stage],
        listeners=listeners,
        # The hook used for evaluation
        hooks=hooks,
    )

    # the pre-task initialization
    # initialize the splade model with the learned weight
    learner_init_tasks = [splade_init_tasks[0]]
    # load the original final projector
    if not cfg.colbert.from_scratch:
        learner_init_tasks.append(
            ColBERTProjectionInitialization(
                hf_id=cfg.colbert.hf_id,
                model=colbert.encoder.encoder.model,
                num_add_tokens=cfg.colbert.doc_additional_tokens,
            ),
        )

    # launch the learner
    outputs = learner.submit(
        launcher=launcher_learner,
        init_tasks=learner_init_tasks,
    )
    helper.tensorboard_service.add(learner, learner.logpath)

    # ----- Evaluations
    # Do the evaluation with a set of thresholds
    dlen = (
        cfg.colbert.tokenize_option.dlen
        if cfg.evaluation.evaluate_in_domain
        else cfg.colbert.tokenize_option.dlen_OoD
    )
    colbert_evaluations = []  # list of evaluation models
    for n_t in cfg.evaluation.pruning.evaluation_norm_threshold:
        colbert_evaluation = ColBERTWithProjectorNormPruning(
            encoder=colbert_document_encoder,
            query_encoder=colbert_query_encoder,
            similarity=DotProductSimilarity(),
            qlen=cfg.colbert.tokenize_option.qlen,
            dlen=dlen,
            mask_attend=cfg.colbert.mask_attend,
            mask_punctuation=True,
            num_add_tokens=cfg.colbert.doc_additional_tokens,
            norm_threshold=n_t,
        ).tag("n_t", n_t)
        colbert_evaluations.append(colbert_evaluation)
    for lp_t in cfg.evaluation.pruning.evaluation_LP_ratio_threshold:
        colbert_evaluation = ColBERTWithLPPruning(
            encoder=colbert_document_encoder,
            query_encoder=colbert_query_encoder,
            similarity=DotProductSimilarity(),
            qlen=cfg.colbert.tokenize_option.qlen,
            dlen=dlen,
            mask_attend=cfg.colbert.mask_attend,
            mask_punctuation=True,
            num_add_tokens=cfg.colbert.doc_additional_tokens,
            lp_threshold=lp_t,
        ).tag("lp_t", lp_t)
        colbert_evaluations.append(colbert_evaluation)
    assert len(colbert_evaluations) > 0, "No pruning strategy provided!"

    if cfg.evaluation.using_best_validate:
        validation_output = outputs.listeners[validation_rerank_splade.id]
        load_models = {}
        if len(cfg.evaluation.evaluate_metric_names) > 0:
            for name in cfg.evaluation.evaluate_metric_names:
                load_models[name] = validation_output[name].tag("cp", name)
        else:  # we evaluate all!
            for name, loaded_model in validation_output.items():
                load_models[name] = loaded_model.tag("cp", name)
    else:
        load_models = {"last_cp": outputs.learned_model.tag("cp", "last")}

    # define the evaluation method
    def evaluate(evaluate_set, first_stage_retriever, load_model, model_id, launcher):
        _, cached_retrievers = evaluate_set.evaluate_first_stage_retriever(
            retriever=first_stage_retriever,
            launcher=launcher,
            model_id=None,
            init_tasks=[
                splade_init_tasks[0],
            ],
        )
        for colbert_evaluation in colbert_evaluations:
            second_stage_retrievers = []
            for cached_retriever in cached_retrievers:
                second_stage_retrievers.append(
                    colbert_evaluation.getRetriever(
                        retriever=cached_retriever,
                        batch_size=cfg.evaluation.retrieval.batch_size,
                        batcher=PowerAdaptativeBatcher(),
                        device=device,
                    ),
                )
            # evaluate on the second stage
            evaluate_set.evaluate_second_stage_retriever(
                retrievers=second_stage_retrievers,
                launcher=launcher,
                model_id="colbert" + model_id,
                init_tasks=[
                    load_model,
                ],
            )

    # store the evaluation set together with the get metric
    total_dataset = []
    if cfg.evaluation.evaluate_in_domain:
        # the in domain retriever
        in_domain_splade_retriever_test = SparseRetriever(
            index=sparse_index,
            topk=cfg.evaluation.retrieval.k,
            batchsize=1,
            encoder=spladev2._query_encoder,
            in_memory=True,
        )
        in_domain_test_set = msmarco_v1_tests()
        for model_id, loaded_model in load_models.items():
            evaluate(
                in_domain_test_set,
                in_domain_splade_retriever_test,
                loaded_model,
                model_id,
                launcher_evaluate,
            )

        total_dataset.append((in_domain_test_set, "in_domain"))
    else:
        cfg.evaluation.ood_set.check_dataset()
        for corpus_name, corpus_id, test_data_id in zip(
            cfg.evaluation.ood_set.name,
            cfg.evaluation.ood_set.documents,
            cfg.evaluation.ood_set.test,
        ):
            corpus = prepare_collection(corpus_id)
            corpus_sparse_index = (
                SparseRetrieverIndexBuilder(
                    batch_size=512,
                    batcher=PowerAdaptativeBatcher(),
                    encoder=spladev2.encoder,
                    device=device,
                    documents=corpus,
                    ordered_index=False,
                    max_docs=cfg.indexation.max_docs,
                )
                .tag("dataset", corpus_name)
                .submit(launcher=launcher_learner, init_tasks=splade_init_tasks)
            )
            corpus_splade_retriever = SparseRetriever(
                index=corpus_sparse_index,
                topk=cfg.evaluation.retrieval.k,
                batchsize=1,
                encoder=spladev2._query_encoder,
                in_memory=True,
            )
            OoD_test_set = OoD_tests(test_data_id)
            for model_id, loaded_model in load_models.items():
                evaluate(
                    OoD_test_set,
                    corpus_splade_retriever,
                    loaded_model,
                    model_id,
                    launcher_evaluate,
                )
            total_dataset.append((OoD_test_set, "out_of_domain"))

    # output the evaluate output
    for dataset, tag in total_dataset:
        if tag == "out_of_domain":
            dataset.output_results_per_tag(printed_metric=["nDCG@10", "Success@5"])
        elif tag == "in_domain":
            dataset.output_results_per_tag(printed_metric=["RR@10", "nDCG@10", "P@20"])

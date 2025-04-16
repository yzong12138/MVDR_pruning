Towards Lossless Token Pruning in Late-Interaction Retrieval Models
===

This repository contains all the details, configurations and scripts necessary in order to reproduce the experiments led in the coontext of the paper *Towards Lossless Token Pruning in Late-Interaction Retrieval Models* by Yuxuan Zong and Benjamin Piwowarski, that was accepted at SIGIR 2025.

## Table of Contents

* [Installation](#installation)
* [Examples](#examples)
* [Contact](#contact)
* [Citation](#citation)

## Installation

The repository is build based on the [experimaestro-ir](https://github.com/experimaestro/experimaestro-ir) framework, which is a repository providing many basic key components in doing IR experiments, including learning, indexing, evaluating, etc. Our dataset access is done under the [ir-datasets](https://github.com/allenai/ir_datasets/) package, through the interface of [datamaestro](https://github.com/experimaestro/datamaestro_text).

The repository can be installed locally by cloning it before running `pip install -r requirements.txt` (Note that in order to get everything to work you might need to ponctually install other libs).

## Examples
Following the [experimaestro framework](https://experimaestro-python.readthedocs.io/en/latest/), the whole pipeline of indexing, training, evaluating are defined under *Experiment* scripts, controlled by configuration file written in YAML. Examples of such files are available under the [experiment](experiments) folder of the repository.

To launch the experiment use the following command:
```unix
 experimaestro run-experiment /path/to/the/config/file --workdir /your/working/directory/
```
Alternatively, if you want to make sure that everything is setup correctly before starting your experiment, you can add the option `--run-mode dry-run` to the command above (Go over the experimental plan without launching the task).

## Experiment Details

For the hyperparameters of the model, please find the details in the `.yaml` files. We show several important ones in the following part:

### Regularization Hyperparameters
The regularization hyperparameters are defined under `RegularizationCoeffConfiguration` class of the `experiments/configuration.py` file. The one used for the experiments are defined under the corresponding hierarchy in the `.yaml` file.

### Pruning Hyperparameters
The pruning hyperparameters are defined under `PruningConfig` class of the `experiments/configuration.py` file. The one used for the experiments are defined under the corresponding hierarchy in the `.yaml` file. We use lists to represent the possible pruning threshold in order to gather several different experiments in the same experimental file.

### Dataset

#### Distillation Samples

Please replace the `distil_data_path` in the `.yaml` to the path where where you store the distillation samples provided by the [ColBERTv2](https://github.com/stanford-futuredata/ColBERT). You can download it by:

```unix
wget https://huggingface.co/colbert-ir/colbertv2.0_msmarco_64way/resolve/main/examples.json?download=true
```

#### Evaluation
We make use of the package `ir-datasets` together with `datamaestro` to facilitate the access of the datasets in our experiments. In `.yaml` file, we only an example of using the *TREC-COVID* and *LoTTE-recreation* for the out-of-domain evaluation and commented out the other dataset used in our experiments.

To use other dataset, please use the following command to check the availability:
```unix
datamaestro search <dataset_name>
```

## Contact

Please feel free to email Yuxuan or the academic' supervisors Benjamin at (name).(surname)@isir.upmc.fr

## Citation

Coming soon...

# SentenceTransformer training script based on,
# https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/nli/training_nli_v2.py

# Python
import io
import os
import math
import random
import logging
import requests
from typing import Optional, List

# Data processing
import pandas as pd
from datasets import load_dataset
from datasets.arrow_dataset import concatenate_datasets

# Transformers
from sentence_transformers import models, losses, datasets
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    LoggingHandler,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
STS_URLS = {
    "train": "https://raw.githubusercontent.com/emrecncelik/sts-benchmark-tr/main/sts-train-tr.csv",
    "dev": "https://raw.githubusercontent.com/emrecncelik/sts-benchmark-tr/main/sts-dev-tr.csv",
    "test": "https://raw.githubusercontent.com/emrecncelik/sts-benchmark-tr/main/sts-test-tr.csv",
}


def download_github_dataset(dataset_url: str):
    dataset_file = requests.get(dataset_url).content
    dataset = pd.read_csv(io.StringIO(dataset_file.decode("utf-8")))
    return dataset


class STransformerNLITrainer:
    def __init__(
        self,
        checkpoint: str,
        output_dir: Optional[str] = "",
        datasets: List[str] = ["mnli", "snli"],
        max_seq_length: int = 75,
        epochs: int = 1,
        batch_size: int = 32,
        max_train_examples: Optional[int] = None,
        max_eval_examples: Optional[int] = None,
    ) -> None:
        # User inputs
        self.checkpoint = checkpoint
        self.output_dir = output_dir
        self.datasets = datasets
        self.max_train_examples = max_train_examples
        self.max_eval_examples = max_eval_examples
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.epochs = epochs

        # Model related
        self.model = None
        self.label2id = None
        self.id2label = None

        # Data related
        self.train_data = {}
        self.train_samples = []
        self.dev_samples = []
        self.test_samples = []
        self.all_nli = None
        self.sts_validation = None
        self.sts_test = None

        if not output_dir:
            self.output_dir = "_".join(
                [self.checkpoint.replace("/", "_"), "stransformer"]
            )
        else:
            dir = "_".join([checkpoint.replace("/", "_"), "stransformer"])
            self.output_dir = os.path.join(self.output_dir, dir)

        self.prepare_for_training()

    def _add_to_samples(self, sent1: str, sent2: str, label: str):
        if sent1 not in self.train_data:
            self.train_data[sent1] = {
                "contradiction": set(),
                "entailment": set(),
                "neutral": set(),
            }
        self.train_data[sent1][label].add(sent2)

    def load_datasets(self):
        logger.info("===================== Loading datasets =====================")
        # NLI
        nli_datasets = []

        if "snli" in self.datasets:
            nli_datasets.append(load_dataset("nli_tr", "snli_tr")["train"])
        if "multinli" in self.datasets:
            nli_datasets.append(load_dataset("nli_tr", "multinli_tr")["train"])

        # Merge NLI datasets
        if len(self.datasets) >= 2:
            self.all_nli = concatenate_datasets(nli_datasets)
        else:
            self.all_nli = nli_datasets[0]

        # Set max train examples
        if self.max_train_examples:
            self.all_nli = self.all_nli.select(range(self.max_train_examples))

        # STS
        self.sts_test = download_github_dataset(STS_URLS["test"])
        self.sts_dev = download_github_dataset(STS_URLS["dev"])

        if self.max_eval_examples:
            self.sts_dev = self.sts_dev.iloc[: self.max_eval_examples, :]

        # Match label ids from dataset
        labels = self.all_nli.features["label"].int2str(range(3))
        self.id2label = {id_: label for id_, label in enumerate(labels)}
        self.label2id = {label: id_ for id_, label in self.id2label.items()}

    def load_model(self):
        logger.info("===================== Loading model =====================")
        word_embedding_model = models.Transformer(
            self.checkpoint, max_seq_length=self.max_seq_length
        )
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(), pooling_mode="mean"
        )
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def prepare_data(self):
        logger.info("===================== Preparing data =====================")

        # Filter -1 labels and convert ids to label names
        self.all_nli = self.all_nli.filter(lambda example: example["label"] != -1)

        for example in self.all_nli:
            sent1 = example["premise"].strip()
            sent2 = example["hypothesis"].strip()

            self._add_to_samples(sent1, sent2, self.id2label[example["label"]])
            self._add_to_samples(sent2, sent1, self.id2label[example["label"]])

        for sent1, others in self.train_data.items():
            if len(others["entailment"]) > 0 and len(others["contradiction"]) > 0:
                self.train_samples.append(
                    InputExample(
                        texts=[
                            sent1,
                            random.choice(list(others["entailment"])),
                            random.choice(list(others["contradiction"])),
                        ]
                    )
                )
                self.train_samples.append(
                    InputExample(
                        texts=[
                            random.choice(list(others["entailment"])),
                            sent1,
                            random.choice(list(others["contradiction"])),
                        ]
                    )
                )
        logging.info(f"\tTrain samples: {len(self.train_samples)}")

        # Development dataset
        for row in self.sts_dev.iterrows():
            score = float(row[1]["score"]) / 5  # Normalize
            self.dev_samples.append(
                InputExample(
                    texts=[row[1]["sentence1_tr"], row[1]["sentence2_tr"]], label=score
                )
            )

        logging.info(f"\tDev samples: {len(self.dev_samples)}")

    def prepare_for_training(self):
        logger.info(
            "===================== Preparing for training ====================="
        )
        self.load_datasets()
        self.load_model()
        self.prepare_data()

    def train(self):
        logger.info("===================== Running Training =====================")
        # Avoids duplicates in a batch
        train_dataloader = datasets.NoDuplicatesDataLoader(
            self.train_samples, batch_size=self.batch_size
        )
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            self.dev_samples, batch_size=self.batch_size, name="sts-dev"
        )
        warmup_steps = math.ceil(len(train_dataloader) * self.epochs * 0.1)
        logger.info(f"Warmup steps: {warmup_steps}")
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=dev_evaluator,
            epochs=self.epochs,
            evaluation_steps=int(len(train_dataloader) * 0.1),
            warmup_steps=warmup_steps,
            output_path=self.output_dir,
            use_amp=False,  # Set to True, if your GPU supports FP16 operations
        )

    def evaluate(self, model_dir: str = None):
        logger.info("===================== Running evaluation =====================")
        logger.info("\tPreparing test data")
        for row in self.sts_test.iterrows():
            score = float(row[1]["score"]) / 5  # Normalize
            self.test_samples.append(
                InputExample(
                    texts=[row[1]["sentence1_tr"], row[1]["sentence2_tr"]], label=score
                )
            )
        logging.info(f"\tTest samples: {len(self.test_samples)}")

        if not model_dir:
            model_dir = self.output_dir

        logger.info(f"\tLoading SentenceTransformer from: {model_dir}")
        model = SentenceTransformer(model_dir)
        test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            self.test_samples, batch_size=self.batch_size
        )
        test_evaluator(model, output_path=model_dir)


if __name__ == "__main__":
    trainer = STransformerNLITrainer(
        checkpoint="dbmdz/bert-base-turkish-cased",
        epochs=10,
        batch_size=128,
    )
    trainer.train()
    trainer.evaluate()
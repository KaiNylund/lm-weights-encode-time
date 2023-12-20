#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, Union, Any, List, Tuple
from itertools import chain

import datasets
import numpy as np
from datasets import load_dataset, Value, Features

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback,

)

from transformers.file_utils import (
    cached_property,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_available,
    is_torch_tpu_available,
)


from transformers.models.t5.modeling_t5 import load_tf_weights_in_t5
from transformers.trainer_utils import get_last_checkpoint, EvalLoopOutput, has_length
from transformers.trainer_pt_utils import find_batch_size, nested_detach, nested_concat

from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, matthews_corrcoef
import torch
from overrides import overrides
from torch import nn

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import DistributedSampler, RandomSampler

from transformers import PreTrainedModel, Trainer
from transformers.integrations import is_fairscale_available
from transformers.models.fsmt.configuration_fsmt import FSMTConfig
from transformers.optimization import (
    Adafactor,
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers.trainer_pt_utils import get_tpu_sampler
from transformers.training_args import ParallelMode
from transformers.utils import is_torch_tpu_available
from transformers import DataCollatorForLanguageModeling
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftConfig, PeftModel


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.13.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

logger = logging.getLogger(__name__)




from transformers import Seq2SeqTrainer

class MT5Trainer(Seq2SeqTrainer):
    @overrides
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.
        Subclass and override to inject custom behavior.
        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)


    @overrides
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)


        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            

            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

            if is_torch_tpu_available():
                xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None
        
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)
        
        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    state_dict: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to saved state_dict to load into model."
        },
    )
    tf_checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory of Tensorflow checkpoint to load into model."
        },
    )
    parallelize: bool = field(
        default=False,
        metadata={
            "help": "Distribute model parameters across multiple GPUs."
        },
    )
    models_to_merge: str = field(
        default=None,
        metadata={
            "help": "Filenames of models to load for merging."
        },
    )
    fishers: str = field(
        default=None,
        metadata={
            "help": "Filenames of saved Fishers to load for merging."
        },
    )
    merging_weights: str = field(
        default=None,
        metadata={
            "help": "Weights to use for weighted average merging of models."
        },
    )
    params_to_skip: str = field(
        default=None,
        metadata={
            "help": "Names of parameters to exclude when merging models."
        },
    )
    lora: bool = field(
        default=False,
        metadata={
            "help": "whether or not to use LoRA to reduce the number of trainable params"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    input_column_1: str = field(default=None, metadata={"help": "Label of input text in dataset."})
    input_column_2: str = field(default=None, metadata={"help": "Label of input text in dataset."})

    target_label: str = field(default=None, metadata={"help": "Label of target text in dataset."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file."})
    train_split: Optional[str] = field(default=None, metadata={"help": "The dataset split to use for training, defaults to None."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics on."
        },
    )
    validation_split: Optional[str] = field(default=None, metadata={"help": "The dataset split to use for validation, defaults to None."})
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics on."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )

    train_eval_files: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Evaluation files passed to the Trainer during training via the eval_dataset param"
        },
    )

    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix_1: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    source_prefix_2: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    patience: Optional[int] = field(
        default=3,
        metadata={
            "help": "early stopping patience "
        },
    )
    lm: bool = field(
        default=False,
        metadata={
            "help": "Are you language modeling?"
        },
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the :obj:`decoder_start_token_id`."
            "Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token "
            "needs to be the target language token.(Usually it is the target language token)"
        },
    )
    calculate_pearson: bool = field(
        default=False,
        metadata={
            "help": "Whether to calculate Pearson correlation between labels and predictions when running evaluation."
        },
    )
    calculate_binary_f1: bool = field(
        default=False,
        metadata={
            "help": "Whether to calculate binary F1-score between labels and predictions when running evaluation."
        },
    )


    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        elif self.input_column_1 is None or (self.target_label is None and not self.lm):
            raise ValueError("Need to specify the input label and the target label.")

        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix_1 is None and model_args.model_name_or_path in [
        "google/mt5-small",
        "google/mt5-base",
        "google/mt5-large",
        "google/mt5-3b",
        "google/mt5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is expected, e.g. with "
            "`--source_prefix_1 'translate English to German: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if not os.path.isdir(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    if training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None and \
      (data_args.train_file is None) and \
      (data_args.validation_file is None):
        # Downloading and loading a dataset from the hub.
        raw_datasets = {}
        if data_args.train_split is not None:
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=data_args.train_split,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        if data_args.validation_split is not None:
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=data_args.validation_split,
                use_auth_token=True if model_args.use_auth_token else None,
            )

        # Infer dataset splits if they aren't given
        if (data_args.train_split is None) and (data_args.validation_split is None):
            print("Inferring dataset splits")
            raw_datasets = load_dataset(
                data_args.dataset_name, data_args.dataset_config_name
            )
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    use_auth_token=True if model_args.use_auth_token else None,
                )
                raw_datasets["train"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    use_auth_token=True if model_args.use_auth_token else None,
                )
    else:
        data_files = {}
        #all_datasets = []
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
            #all_datasets.append(load_dataset("json", data_files=data_files["train"]))
        if data_args.train_eval_files is not None:
            for i, fname in enumerate(data_args.train_eval_files):
                data_files["train_eval" + str(i)] = fname

        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
            #all_datasets.append(load_dataset("json", data_files=data_files["validation"]))
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
            #all_datasets.append(load_dataset("json", data_files=data_files["test"]))

        #features = Features({data_args.input_column_1: Value(dtype='string', id=None),
        #            data_args.target_label: Value(dtype='string', id=None)})

        #raw_datasets = load_dataset("json", data_files=data_files, features=features)
        raw_datasets = load_dataset("json", data_files=data_files)


    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    print(model_args.lora, model_args.model_name_or_path, os.path.exists(model_args.model_name_or_path + "/adapter_config.json"))

    if model_args.lora and (model_args.model_name_or_path not in ["t5-small", "t5-large", "t5-3b"]):
        print("Loading pretrained LoRA model...")
        config = PeftConfig.from_pretrained(model_args.model_name_or_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
        if training_args.do_train:
            print("Training...")
            model = PeftModel.from_pretrained(model, model_args.model_name_or_path, config=config, is_trainable=True)
        else:
            model = PeftModel.from_pretrained(model, model_args.model_name_or_path, config=config)
        model.print_trainable_parameters()
    else:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # If we're not loading from a path, convert the vanilla model to peft
    if model_args.lora and (model_args.model_name_or_path in ["t5-small", "t5-large", "t5-3b"]):
        peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # load model parameters from Tensorflow checkpoint
    if model_args.tf_checkpoint_dir is not None:
        logger.info("Loading model from Tensorflow checkpoint directory: "
                    f"{model_args.tf_checkpoint_dir}")
        model = load_tf_weights_in_t5(model, model.config,
                                      model_args.tf_checkpoint_dir)

    # load state_dict after replacing modules
    if model_args.state_dict is not None:
        logger.info(f'Loading parameters from file: {model_args.state_dict}')
        state_dict = torch.load(model_args.state_dict, map_location='cpu')
        model.load_state_dict(state_dict)

    # distribute model parameters across GPUs
    if model_args.parallelize:
        model.parallelize()

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # merge models and load resulting state_dict into current model
    if model_args.models_to_merge is not None:
        models = [torch.load(elem, map_location='cpu')
                  for elem in model_args.models_to_merge.split(',')]

        fishers, weights, skip = None, None, list()
        if model_args.fishers is not None:
            fishers = [torch.load(elem, map_location='cpu')
                       for elem in model_args.fishers.split(',')]
        if model_args.merging_weights is not None:
            weights = list(map(float, model_args.merging_weights.split(',')))
        if model_args.params_to_skip is not None:
            skip = model_args.params_to_skip.split(',')

        merged_state_dict = merge_models(models, weights=weights,
                                         fishers=fishers, skip=skip)
        model.load_state_dict(merged_state_dict)

    prefix_1 = data_args.source_prefix_1 if data_args.source_prefix_1 is not None else ""
    prefix_2 = data_args.source_prefix_2 if data_args.source_prefix_2 is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the label names for input/target.
    input_column_1 = data_args.input_column_1
    input_column_2 = data_args.input_column_2
    
    target_label = data_args.target_label

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False
    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def group_texts(examples, block_size):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    

    def lm_preprocess_function(examples):
        premises = [ex for ex in examples[input_column_1]]
        if input_column_2 and input_column_2 != "None":
            hypotheses = [ex for ex in examples[input_column_2]]
            inputs = [f"{prefix_1} " + x + f" {prefix_2} " + y for x,y in zip(premises, hypotheses)]
        else:
            inputs = [prefix_1 + x for x in premises]
        model_inputs = tokenizer(inputs)
        model_inputs = group_texts(model_inputs, data_args.max_source_length)
        source = {key: model_inputs[key][:-1] for key in model_inputs.keys()}
        targets = {key: model_inputs[key][1:] for key in model_inputs.keys()}
        result = {'input_ids': source['input_ids'], 'attention_mask': source['attention_mask'], 'labels': targets['input_ids']}
        return result
    
    def supervised_preprocess_function(examples):
        premises = [ex for ex in examples[input_column_1]]
        if input_column_2 and input_column_2 != "None":
            hypotheses = [ex for ex in examples[input_column_2]]
        #targets = ["negative" if ex == 0 else "positive" for ex in examples[target_label]]
        targets = [str(ex) for ex in examples[target_label]]
        if input_column_2 and input_column_2 != "None":
            inputs = [f"{prefix_1} " + x + f" {prefix_2} " + y for x,y in zip(premises, hypotheses)]
        else:
            inputs = [prefix_1 + x for x in premises]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        #print(train_dataset[0:10])
        #print(data_args.max_train_samples)
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_eval_datasets = None
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                lm_preprocess_function if data_args.lm else supervised_preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            #print(train_dataset[0:10]["labels"])

            if data_args.train_eval_files is not None:
                train_eval_datasets = {}
                training_args.evaluation_strategy = "steps"
                training_args.eval_steps = 200
                for i, fname in enumerate(data_args.train_eval_files):
                    train_eval_datasets[fname] = raw_datasets["train_eval" + str(i)].map(
                        lm_preprocess_function if data_args.lm else supervised_preprocess_function,
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        remove_columns=column_names,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Running tokenizer on train eval datasets",
                    )
    
    if  training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                lm_preprocess_function if data_args.lm else supervised_preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                lm_preprocess_function if data_args.lm else supervised_preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # accuracy (exact string match)
    def accuracy_metric(predictions, labels):
        return np.mean([pred == lab for pred, lab in zip(predictions, labels)])

    # general f1 score function
    def f1_metric(predictions, labels, average='macro'):
        # create mapping from unique set of outputs to integer labels
        uniq = sorted(list(set(predictions).union(set(labels))))
        mapping = {k : v for v, k in enumerate(uniq)}
        predictions = [mapping[pred] for pred in predictions]
        labels = [mapping[lab] for lab in labels]
        return f1_score(labels, predictions, average=average)

    # pearson correlation with safe casting to float
    def pearsonr(predictions, labels):
        def safe_float(x):
            try:
                return float(x)
            except ValueError:
                return 0
        predictions = [safe_float(elem) for elem in predictions]
        labels = [safe_float(elem) for elem in labels]
        return np.corrcoef(predictions, labels)[0, 1]

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        return preds, labels

    
    def compute_metrics_predict_with_generate(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        accuracy = accuracy_metric(decoded_labels, decoded_preds)
        macro_f1 = f1_metric(decoded_labels, decoded_preds, average='macro')
        micro_f1 = f1_metric(decoded_labels, decoded_preds, average='micro')
        matthews_cc = matthews_corrcoef(decoded_labels, decoded_preds)
        result = {"accuracy": accuracy, "macro_f1" : macro_f1,
                  "micro_f1" : micro_f1, "matthews_corrcoef" : matthews_cc}
        if data_args.calculate_pearson:
            result["pearson_corr"] = pearsonr(decoded_labels, decoded_preds)
        if data_args.calculate_binary_f1:
            result["binary_f1"] = f1_metric(decoded_labels, decoded_preds)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        if data_args.dataset_name == 'paws-x':
            LABEL_MAP = {(0,):[333], (1,): [497]}
        elif data_args.dataset_name == 'xnli':
            LABEL_MAP = {(0,): [356], (1,):[333], (2,):[497]}
        elif data_args.dataset_name == 'amazon_polarity':
            LABEL_MAP = {(0,): [1465], (1,): [2841]}
        elif data_args.dataset_name == 'sst2':
            LABEL_MAP = {(0,): [1465], (1,): [2841]}
        # assert all(len(LABEL_MAP[x]) == len(x) for x in LABEL_MAP)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
        all_losses = []
        # loop through labels, get per label losses for each example
        for key, label in LABEL_MAP.items():
            preds = torch.Tensor(preds)
            if not isinstance(label, list):
                label = [label]
            labels_ = torch.Tensor([ label + [tokenizer.pad_token_id] for _ in range(preds.shape[0])])
            loss = loss_fn(preds.view(-1, preds.shape[-1]), labels_.view(-1).long())
            all_losses.append(loss.reshape(labels.shape).unsqueeze(-1))
        losses = torch.cat(all_losses, -1)
        # the prediction is the label that gets minimum loss
        preds = torch.argmin(losses, -1).cpu().numpy()
        # the last token in each case in </s>, ignore that in label map
        preds = [LABEL_MAP[tuple(row[:-1])] for row in preds]
        


        
        # # map labels to token index
        # if data_args.dataset_name == 'paws-x':
        #     LABEL_MAP = {0:333, 1:497}
            
        # elif data_args.dataset_name == 'xnli':
        #     LABEL_MAP = {0: 356, 1:333, 2:497}

        # loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
        # all_losses = []
        # # loop through labels, get per label losses for each example
        # for key, label in LABEL_MAP.items():
        #     preds = torch.Tensor(preds)
        #     labels_ = torch.Tensor([[label, tokenizer.pad_token_id] for _ in range(preds.shape[0])])
        #     loss = loss_fn(preds.view(-1, preds.shape[-1]), labels_.view(-1).long())
        #     all_losses.append(loss.reshape(labels.shape).unsqueeze(-1))
        # losses = torch.cat(all_losses, -1)
        # # the prediction is the label that gets minimum loss
        # preds = torch.argmin(losses, -1).cpu().numpy()
        # # the last token in each case in </s>, ignore that in label map
        # preds[:, 0] = [LABEL_MAP[x] for x in preds[:, 0]]
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        accuracy = accuracy_metric(decoded_labels, decoded_preds)
        macro_f1 = f1_metric(decoded_labels, decoded_preds, average='macro')
        micro_f1 = f1_metric(decoded_labels, decoded_preds, average='micro')
        matthews_cc = matthews_corrcoef(decoded_labels, decoded_preds)
        result = {"accuracy": accuracy, "macro_f1" : macro_f1,
                  "micro_f1" : micro_f1, "matthews_corrcoef" : matthews_cc}
        if data_args.calculate_pearson:
            result["pearson_corr"] = pearsonr(decoded_labels, decoded_preds)
        if data_args.calculate_binary_f1:
            result["binary_f1"] = f1_metric(decoded_labels, decoded_preds)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    if data_args.patience is not None:
        early_stopping = EarlyStoppingCallback(early_stopping_patience=data_args.patience)
        
    # Initialize our Trainer
    if data_args.lm:
        cm = None
    elif training_args.predict_with_generate:
        cm = compute_metrics_predict_with_generate
    else:
        cm = compute_metrics

    #key_list = [key for key, _ in model.named_modules()]
    #print(key_list)

    # Training
    if training_args.do_train:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=train_eval_datasets,
            tokenizer=tokenizer,
            data_collator=data_collator,
            # callbacks=[early_stopping] if data_args.patience is not None else None,
            compute_metrics=cm,
        )

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        #trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.model.save_pretrained(trainer.args.output_dir)
        tokenizer.save_pretrained(trainer.args.output_dir)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        if model_args.lora:
            eval_model = model.merge_and_unload()
        else:
            eval_model = model
        trainer = Seq2SeqTrainer(
            model=eval_model,
            args=training_args,
            train_dataset=None,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            # callbacks=[early_stopping] if data_args.patience is not None else None,
            compute_metrics=cm,
        )
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        if model_args.lora:
            eval_model = model.merge_and_unload()
        else:
            eval_model = model
        trainer = Seq2SeqTrainer(
            model=eval_model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            # callbacks=[early_stopping] if data_args.patience is not None else None,
            compute_metrics=cm,
        )
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="test", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["test_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "classification"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

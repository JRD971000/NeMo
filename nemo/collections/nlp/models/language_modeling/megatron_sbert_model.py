# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import itertools
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.trainer.trainer import Trainer
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from nemo.collections.nlp.data.language_modeling.megatron import dataset_utils
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.models.language_modeling.megatron.bert_model import (
    BertLMHead,
    BertModel,
    bert_extended_attention_mask,
    post_language_model_processing,
)
from nemo.collections.nlp.models.language_modeling.megatron_base_model import MegatronBaseModel
from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel
from nemo.collections.nlp.modules.common.megatron.build_model import build_model
from nemo.collections.nlp.modules.common.megatron.language_model import get_language_model
from nemo.collections.nlp.modules.common.megatron.module import Float16Module, MegatronModule
from nemo.collections.nlp.modules.common.megatron.transformer import get_layer_norm
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    average_losses_across_data_parallel_group,
    build_position_ids,
    erf_gelu,
    get_linear_layer,
    get_params_for_weight_decay_optimization,
    init_method_normal,
    openai_gelu,
    parallel_lm_logits,
    scaled_init_method_normal,
)
from nemo.collections.nlp.parts.nlp_overrides import GradScaler
from nemo.collections.nlp.parts.utils_funcs import get_last_rank
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.neural_types import ChannelType, MaskType, NeuralType
from nemo.utils import AppState, logging

try:
    from apex.transformer.enums import AttnMaskType
    from apex.transformer.pipeline_parallel.utils import get_num_microbatches
    from apex.transformer.tensor_parallel.layers import set_tensor_model_parallel_attributes

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

    # fake missing classes with None attributes
    AttnMaskType = ApexGuardDefaults()

try:
    from megatron.core import ModelParallelConfig, parallel_state, tensor_parallel

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    ModelParallelConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False

try:
    import logging

    from lddl.torch_mp import get_bert_pretrain_data_loader

    HAVE_LDDL = True
except (ImportError, ModuleNotFoundError):
    HAVE_LDDL = False

try:
    from megatron.core import parallel_state
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.transformer.module import Float16Module as MCoreFloat16Module
    from megatron.core.transformer.transformer_config import TransformerConfig

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    TransformerConfig = ApexGuardDefaults
    HAVE_MEGATRON_CORE = False

import numpy as np
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


class MultiplePositivesNegativesDataset(Dataset):
    """SentenceTransformer tokenizer and MultipleNegativesRankingLoss expects
        a single positive and a single hard-negative (optional) per example.
        This Dataset manages the case where there is more than one positive or negative
        available, in form of a list.
        It uses the list of positives/negatives as a queue, where for each epoch the 
        first positive/negative of the queue is used for training, after which the
        item is moved to the end of the queue.
        If num_hard_negs > 1, multiple negatives will be sampled for each example.

        Args:
            data (List[Dict[str, str]]): A list of Dict whose 
            keys are "question", "pos_doc", "neg_doc"
            num_hard_negs (int): Number of hard-negatives for each query to sample
            shuffled_negs (bool, optional): Whether the negatives per example
            needs to be shuffled in the initialization. Defaults to False.
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        shuffled_negs: bool = False,
        num_hard_negs: int = 1,
        query_prefix: str = "",
        passage_prefix: str = "",
    ):
        self.data = data
        self.num_hard_negs = num_hard_negs
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

        if shuffled_negs:
            for example in self.data:
                random.shuffle(example["neg_doc"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        example = self.data[item]
        question = f'{self.query_prefix} {example["question"]}'.strip()
        texts = [question]

        positive = example["pos_doc"]
        if isinstance(positive, list):
            # Dequeues one positive and adds it at end of the queue
            positive = example["pos_doc"].pop(0)
            example["pos_doc"].append(positive)

        positive = f"{self.passage_prefix} {positive}".strip()
        texts.append(positive)

        negative = []
        if "neg_doc" in example:
            negative = example["neg_doc"]
            selected_negs = []
            if isinstance(negative, list):
                for _ in range(self.num_hard_negs):
                    if len(example["neg_doc"]) > 0:
                        # Dequeues a negative and adds it at end of the queue
                        negative = example["neg_doc"].pop(0)
                        selected_negs.append(negative)
                        example["neg_doc"].append(negative)
                    else:
                        # Providing empty hard-negative, for this example,
                        # so that it matches the number of hard negatives
                        # of the other examples
                        selected_negs.append("")

            else:
                selected_negs = [negative]
            selected_negs = [f"{self.passage_prefix} {neg}".strip() for neg in selected_negs]
            texts.extend(selected_negs)
        return texts


##########################
# Below class is copied from SentenceTransformer library: https://github.com/UKPLab/sentence-transformers/blob/08a57b4a19ddaf7cccda51cd0c2c8af7bbc339a3/sentence_transformers/models/Normalize.py
##########################


class Normalize(nn.Module):
    """
    This layer normalizes embeddings to unit length
    """

    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, features: Dict[str, Tensor]):
        features.update({"sentence_embedding": F.normalize(features["sentence_embedding"], p=2, dim=1)})
        return features


##########################
# Below class is copied from SentenceTransformer library: https://github.com/UKPLab/sentence-transformers/blob/08a57b4a19ddaf7cccda51cd0c2c8af7bbc339a3/sentence_transformers/models/Pooling.py
##########################


class Pooling(nn.Module):
    """Performs pooling (max or mean) on the token embeddings.

    Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding. This layer also allows to use the CLS token if it is returned by the underlying word embedding model.
    You can concatenate multiple poolings together.

    :param word_embedding_dimension: Dimensions for the word embeddings
    :param pooling_mode: Can be a string: mean/max/cls. If set, overwrites the other pooling_mode_* settings
    :param pooling_mode_cls_token: Use the first token (CLS token) as text representations
    :param pooling_mode_max_tokens: Use max in each dimension over all tokens.
    :param pooling_mode_mean_tokens: Perform mean-pooling
    :param pooling_mode_mean_sqrt_len_tokens: Perform mean-pooling, but divide by sqrt(input_length).
    :param pooling_mode_weightedmean_tokens: Perform (position) weighted mean pooling, see https://arxiv.org/abs/2202.08904
    :param pooling_mode_lasttoken: Perform last token pooling, see https://arxiv.org/abs/2202.08904 & https://arxiv.org/abs/2201.10005
    """

    def __init__(
        self,
        word_embedding_dimension: int,
        pooling_mode: str = None,
        pooling_mode_cls_token: bool = False,
        pooling_mode_max_tokens: bool = False,
        pooling_mode_mean_tokens: bool = True,
        pooling_mode_mean_sqrt_len_tokens: bool = False,
        pooling_mode_weightedmean_tokens: bool = False,
        pooling_mode_lasttoken: bool = False,
    ):
        super(Pooling, self).__init__()

        self.config_keys = [
            "word_embedding_dimension",
            "pooling_mode_cls_token",
            "pooling_mode_mean_tokens",
            "pooling_mode_max_tokens",
            "pooling_mode_mean_sqrt_len_tokens",
            "pooling_mode_weightedmean_tokens",
            "pooling_mode_lasttoken",
        ]

        if pooling_mode is not None:  # Set pooling mode by string
            pooling_mode = pooling_mode.lower()
            assert pooling_mode in ["mean", "max", "cls", "weightedmean", "lasttoken"]
            pooling_mode_cls_token = pooling_mode == "cls"
            pooling_mode_max_tokens = pooling_mode == "max"
            pooling_mode_mean_tokens = pooling_mode == "mean"
            pooling_mode_weightedmean_tokens = pooling_mode == "weightedmean"
            pooling_mode_lasttoken = pooling_mode == "lasttoken"

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
        self.pooling_mode_weightedmean_tokens = pooling_mode_weightedmean_tokens
        self.pooling_mode_lasttoken = pooling_mode_lasttoken

        pooling_mode_multiplier = sum(
            [
                pooling_mode_cls_token,
                pooling_mode_max_tokens,
                pooling_mode_mean_tokens,
                pooling_mode_mean_sqrt_len_tokens,
                pooling_mode_weightedmean_tokens,
                pooling_mode_lasttoken,
            ]
        )
        self.pooling_output_dimension = pooling_mode_multiplier * word_embedding_dimension

    def __repr__(self):
        return "Pooling({})".format(self.get_config_dict())

    def get_pooling_mode_str(self) -> str:
        """
        Returns the pooling mode as string
        """
        modes = []
        if self.pooling_mode_cls_token:
            modes.append("cls")
        if self.pooling_mode_mean_tokens:
            modes.append("mean")
        if self.pooling_mode_max_tokens:
            modes.append("max")
        if self.pooling_mode_mean_sqrt_len_tokens:
            modes.append("mean_sqrt_len_tokens")
        if self.pooling_mode_weightedmean_tokens:
            modes.append("weightedmean")
        if self.pooling_mode_lasttoken:
            modes.append("lasttoken")

        return "+".join(modes)

    def forward(self, features: Dict[str, Tensor]):
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"]

        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            cls_token = features.get("cls_token_embeddings", token_embeddings[:, 0])  # Take first token by default
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if "token_weights_sum" in features:
                sum_mask = features["token_weights_sum"].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))
        if self.pooling_mode_weightedmean_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            # token_embeddings shape: bs, seq, hidden_dim
            weights = (
                torch.arange(start=1, end=token_embeddings.shape[1] + 1)
                .unsqueeze(0)
                .unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
                .to(token_embeddings.device)
            )
            assert weights.shape == token_embeddings.shape == input_mask_expanded.shape
            input_mask_expanded = input_mask_expanded * weights

            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if "token_weights_sum" in features:
                sum_mask = features["token_weights_sum"].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)
        if self.pooling_mode_lasttoken:
            bs, seq_len, hidden_dim = token_embeddings.shape
            # attention_mask shape: (bs, seq_len)
            # Get shape [bs] indices of the last token (i.e. the last token for each batch item)
            # argmin gives us the index of the first 0 in the attention mask; We get the last 1 index by subtracting 1
            # Any sequence where min == 1, we use the entire sequence length since argmin = 0
            values, indices = torch.min(attention_mask, 1, keepdim=False)
            gather_indices = torch.where(values == 0, indices, seq_len) - 1  # Shape [bs]

            # There are empty sequences, where the index would become -1 which will crash
            gather_indices = torch.clamp(gather_indices, min=0)

            # Turn indices from shape [bs] --> [bs, 1, hidden_dim]
            gather_indices = gather_indices.unsqueeze(-1).repeat(1, hidden_dim)
            gather_indices = gather_indices.unsqueeze(1)
            assert gather_indices.shape == (bs, 1, hidden_dim)

            # Gather along the 1st dim (seq_len) (bs, seq_len, hidden_dim -> bs, hidden_dim)
            # Actually no need for the attention mask as we gather the last token where attn_mask = 1
            # but as we set some indices (which shouldn't be attended to) to 0 with clamp, we
            # use the attention mask to ignore them again
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embedding = torch.gather(token_embeddings * input_mask_expanded, 1, gather_indices).squeeze(dim=1)
            output_vectors.append(embedding)

        output_vector = torch.cat(output_vectors, 1)
        features.update({"sentence_embedding": output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}


class SBertModel(BertModel):
    """
    Bert Language model.
    Model returns [seq, batch, hidden] shape
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        num_layers,
        num_attention_heads,
        ffn_hidden_size,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=True,
        post_process=True,
        init_method_std=0.02,
        fp16_lm_cross_entropy=False,
        hidden_dropout=0.1,
        precision=16,
        fp32_residual_connection=False,
        activations_checkpoint_granularity=None,
        activations_checkpoint_method=None,
        activations_checkpoint_num_layers=1,
        activations_checkpoint_layers_per_pipeline=None,
        layernorm_epsilon=1e-5,
        normalization='layernorm',
        transformer_block_type='pre_ln',
        masked_softmax_fusion=False,
        bias_gelu_fusion=True,
        bias_dropout_add_fusion=True,
        openai_gelu=False,
        onnx_safe=False,
        add_binary_head=True,
        skip_head=False,
        megatron_legacy=False,
        sequence_parallel=False,
        position_embedding_type='learned_absolute',
    ):
        super().__init__(
            config,
            vocab_size,
            hidden_size,
            max_position_embeddings,
            num_layers,
            num_attention_heads,
            ffn_hidden_size,
            apply_query_key_layer_scaling,
            kv_channels,
            num_tokentypes,
            parallel_output,
            pre_process,
            post_process,
            init_method_std,
            fp16_lm_cross_entropy,
            hidden_dropout,
            precision,
            fp32_residual_connection,
            activations_checkpoint_granularity,
            activations_checkpoint_method,
            activations_checkpoint_num_layers,
            activations_checkpoint_layers_per_pipeline,
            layernorm_epsilon,
            normalization,
            transformer_block_type,
            masked_softmax_fusion,
            bias_gelu_fusion,
            bias_dropout_add_fusion,
            openai_gelu,
            onnx_safe,
            add_binary_head,
            skip_head,
            megatron_legacy,
            sequence_parallel,
            position_embedding_type,
        )

        self.pooling_add_on = Pooling(
            word_embedding_dimension=1024,
            pooling_mode_cls_token=False,
            pooling_mode_mean_tokens=True,
            pooling_mode_max_tokens=False,
            pooling_mode_mean_sqrt_len_tokens=False,
        )

        self.normalize_add_on = Normalize()

    def forward(
        self,
        bert_model_input,
        attention_mask,
        token_type_ids=None,
        lm_labels=None,
        checkpoint_activations_all_layers=None,
    ):

        extended_attention_mask = bert_extended_attention_mask(attention_mask)

        if parallel_state.is_pipeline_first_stage():
            input_ids = bert_model_input
            position_ids = build_position_ids(input_ids)
        else:
            position_ids = None
            input_ids = None

        lm_output = self.language_model(
            input_ids,
            position_ids,
            extended_attention_mask,
            token_type_ids=token_type_ids,
            checkpoint_activations_all_layers=checkpoint_activations_all_layers,
        )

        if self.post_process and self.add_binary_head:

            lm_output, pooled_output = lm_output
        else:
            pooled_output = None
                            
        add_on_inputs = {"token_embeddings": lm_output[0].permute(1, 0, 2), "attention_mask": attention_mask}
        lm_output = self.pooling_add_on(add_on_inputs)
        lm_output = self.normalize_add_on(lm_output)

        return lm_output['sentence_embedding']


class MegatronSBertModel(MegatronBertModel):
    """
    Megatron Bert pretraining.
    Model returns [batch, seq, hidden] shape
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):

        super().__init__(cfg, trainer=trainer)
        self.stop_criteria=False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(label_smoothing=cfg.get('label_smoothing', 0.0))
        softmax_temp = cfg.get('softmax_temp', 0.05)
        self.scale = 1.0 / softmax_temp
        train_file_path = self.cfg.data.data_prefix
        with open(train_file_path) as f:
            train_data = json.load(f)

        random_seed=42
        set_seed(random_seed)
        random.shuffle(train_data)

        self.train_data = train_data

    def model_provider_func(self, pre_process, post_process):
        cfg = self.cfg
        num_tokentypes = 2 if cfg.bert_binary_head else 0

        if self.mcore_bert:
            raise ValueError("mcore not supported for SBERT")

        else:
            model = SBertModel(
                config=self.model_parallel_config,
                vocab_size=self.padded_vocab_size,
                hidden_size=cfg.hidden_size,
                max_position_embeddings=cfg.max_position_embeddings,
                num_layers=cfg.num_layers,
                num_attention_heads=cfg.num_attention_heads,
                apply_query_key_layer_scaling=cfg.get('apply_query_key_layer_scaling', True),
                kv_channels=cfg.get('kv_channels', None),
                ffn_hidden_size=cfg.ffn_hidden_size,
                num_tokentypes=num_tokentypes,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
                init_method_std=cfg.get('init_method_std', 0.02),
                fp16_lm_cross_entropy=cfg.get('fp16_lm_cross_entropy', False),
                hidden_dropout=cfg.get('hidden_dropout', 0.1),
                precision=cfg.get('precision', 16),
                fp32_residual_connection=cfg.get('fp32_residual_connection', False),
                activations_checkpoint_granularity=self.cfg.get('activations_checkpoint_granularity', None),
                activations_checkpoint_method=self.cfg.get('activations_checkpoint_method', None),
                activations_checkpoint_num_layers=self.cfg.get('activations_checkpoint_num_layers', 1),
                activations_checkpoint_layers_per_pipeline=self.cfg.get(
                    'activations_checkpoint_layers_per_pipeline', None
                ),
                layernorm_epsilon=cfg.get('layernorm_epsilon', 1e-5),
                masked_softmax_fusion=cfg.get('masked_softmax_fusion', True),
                normalization=cfg.get('normalization', 'layernorm'),
                transformer_block_type=cfg.get('transformer_block_type', 'pre_ln'),
                bias_gelu_fusion=cfg.get('bias_gelu_fusion', True),
                bias_dropout_add_fusion=cfg.get("bias_dropout_add_fusion", True),
                onnx_safe=cfg.get('onnx_safe', False),
                add_binary_head=cfg.bert_binary_head,
                skip_head=cfg.get('skip_head', False),
                megatron_legacy=cfg.get('megatron_legacy', False),
                position_embedding_type=self.cfg.get("position_embedding_type", "learned_absolute"),
            )

        return model

    def build_train_valid_test_datasets(self):

        train_file_path = self.cfg.data.data_prefix

        # with open(train_file_path) as f:
        #     train_data = json.load(f)

        # random_seed=42
        # set_seed(random_seed)
        # random.shuffle(train_data)

        # self.train_data = train_data
        train_data = self.train_data

        
        query_prefix = "query:"
        passage_prefix = "passage:"
        evaluation_sample_size = self.cfg.data.get("evaluation_sample_size", 100)
        hard_negatives_to_train = self.cfg.data.get("hard_negatives_to_train", 4)
        evaluation_steps = self.cfg.data.get("evaluation_steps", 100)

        # TODO @ataghibakhsh: Handle valid and test datasets better

        self._train_ds = None
        self._validation_ds = None
        self._test_ds = None

        if train_file_path:  # we don't support calculating validation loss for multiple train files
            valid_data = None
            if evaluation_sample_size:
                if evaluation_steps == 0:
                    raise ValueError(
                        "The --evaluation_steps should be greater than 0 " "when --evaluation_sample_size is set"
                    )

                if evaluation_sample_size >= len(train_data):
                    raise ValueError("The --evaluation_sample_size cannot be greater " "than train set size.")

                valid_data = train_data[-evaluation_sample_size:]
                train_data = train_data[:-evaluation_sample_size]


            if evaluation_sample_size:
                self._validation_ds = MultiplePositivesNegativesDataset(
                    valid_data,
                    num_hard_negs=hard_negatives_to_train,
                    query_prefix=query_prefix,
                    passage_prefix=passage_prefix,
                )

        self._train_ds = MultiplePositivesNegativesDataset(
            train_data, num_hard_negs=hard_negatives_to_train, query_prefix=query_prefix, passage_prefix=passage_prefix
        )

        if self._train_ds is not None:
            logging.info(f'Length of train dataset: {len(self._train_ds)}')
        if self._validation_ds is not None:
            logging.info(f'Length of val dataset: {len(self._validation_ds)}')
        if self._test_ds is not None:
            logging.info(f'Length of test dataset: {len(self._test_ds)}')
        logging.info(f'Finished building Bert datasets.')

        return self._train_ds, self._validation_ds, self._test_ds

    def setup(self, stage=None):
        """ PTL hook that is executed after DDP spawns.
            We setup datasets here as megatron datasets require DDP to instantiate.
            See https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#setup for more information.
        Args:
            stage (str, optional): Can be 'fit', 'validate', 'test' or 'predict'. Defaults to None.
        """

        num_parameters_on_device, total_num_parameters = self._get_total_params_across_model_parallel_groups_gpt_bert(
            self.model
        )

        logging.info(
            f'Pipeline model parallel rank: {parallel_state.get_pipeline_model_parallel_rank()}, '
            f'Tensor model parallel rank: {parallel_state.get_tensor_model_parallel_rank()}, '
            f'Number of model parameters on device: {num_parameters_on_device:.2e}. '
            f'Total number of model parameters: {total_num_parameters:.2e}.'
        )

        resume_checkpoint_path = self.trainer.ckpt_path
        if resume_checkpoint_path:
            init_consumed_samples = self._extract_consumed_samples_from_ckpt(resume_checkpoint_path)
        else:
            init_consumed_samples = 0
        self.init_consumed_samples = init_consumed_samples
        self.init_global_step = self.trainer.global_step

        if stage == 'predict':
            return
        else:
            # TODO: consider adding a ModelPT guard to check if model is being restored.
            # allowing restored models to optionally setup datasets
            if self.cfg.data.dataloader_type == "LDDL":
                self.build_LDDL_data(self.cfg.data)
                torch.distributed.barrier()
            else:
                self.build_train_valid_test_datasets()
                self.setup_training_data(self.cfg.data)
                self.setup_validation_data(self.cfg.data)
                # self.setup_test_data(self.cfg.data)

        # when using pipeline model parallel the final stage need to initialize word embeddings
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            if isinstance(self.model, list):
                for i, module in enumerate(self.model):
                    parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                    sync_embeddings = (
                        module.initialize_last_stage_with_word_embeddings
                        if self.mcore_bert
                        else module.sync_initial_word_embeddings
                    )
                    sync_embeddings()
                parallel_state.set_virtual_pipeline_model_parallel_rank(0)
            else:
                sync_embeddings = (
                    self.model.initialize_last_stage_with_word_embeddings
                    if self.mcore_bert
                    else self.model.sync_initial_word_embeddings
                )
                sync_embeddings()

        if self.cfg.get('transformer_engine', False) or self.cfg.get('mcore_bert', False):
            self.setup_transformer_engine_tp_groups()

    def build_pretraining_data_loader(self, dataset, consumed_samples):
        """Buld dataloader given an input dataset."""

        if dataset is None:
            return None

        # Megatron sampler
        if hasattr(self.cfg.data, 'dataloader_type') and self.cfg.data.dataloader_type is not None:
            if self.cfg.data.dataloader_type == 'single':
                batch_sampler = MegatronPretrainingSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    global_batch_size=self.cfg.global_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=self.cfg.get('drop_last', True),
                )
            elif self.cfg.data.dataloader_type == 'cyclic':
                batch_sampler = MegatronPretrainingRandomSampler(
                    total_samples=len(dataset),
                    consumed_samples=consumed_samples,
                    micro_batch_size=self.cfg.micro_batch_size,
                    data_parallel_rank=parallel_state.get_data_parallel_rank(),
                    data_parallel_size=parallel_state.get_data_parallel_world_size(),
                    drop_last=self.cfg.get('drop_last', True),
                )
            else:
                raise ValueError('cfg.data.dataloader_type must be "single" or "cyclic"')
        else:
            raise ValueError('cfg.data.dataloader_type not found. Must be "single" or "cyclic"')

        # Torch dataloader.
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            persistent_workers=True if self.cfg.data.num_workers > 0 else False,
        )

        dataloader.collate_fn = self.batching_collate

        return dataloader

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):

        max_seq_length = self.cfg.encoder_seq_length
        do_lower_case = self.cfg.tokenizer.get("do_lower_case", False)
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output["text_keys"] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output["text_keys"].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Lowercase
        if do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(
            self.tokenizer.tokenizer(
                *to_tokenize, padding=True, truncation="longest_first", return_tensors="pt", max_length=max_seq_length,
            )
        )
        return output

    def batching_collate(self, batch):
        """
            Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
            Here, batch is a list of InputExample instances: [InputExample(...), ...]

            :param batch:
                a batch from a SmartBatchingDataset
            :return:
                a batch of tensors for the model
            """
        texts = [example for example in batch]
        sentence_features = [self.tokenize(sentence) for sentence in zip(*texts)]
        return sentence_features

    def training_step(self, dataloader_iter, batch_idx):
        self.in_valid = False
        self.valid_counter = 0
        self.stop_criteria = True
        self._optimizer.zero_grad()

        if self.with_distributed_adam:
            # hack to enable overlapping param sync and forward compute
            # note: the distributed optimizer monkey-patches each
            # parameter's __getattribute__ function so that it can
            # launch parameter all-gathers the first time the
            # parameter is accessed after the optimizer step. However,
            # PyTorch directly passes embedding parameters into a C++,
            # bypassing this process. A quick-and-dirty hack is to
            # manually interact with the parameter.
            modules = self.model if isinstance(self.model, list) else [self.model]
            for module in modules:
                if isinstance(module, (Float16Module, MCoreFloat16Module)):
                    module = module.module
                if not self.mcore_bert:
                    module = module.language_model
                if hasattr(module, 'embedding'):
                    for param in module.embedding.parameters():
                        param.data_ptr()

        if self.cfg.data.dataloader_type == "LDDL":
            # this is of type bert dataset
            seq_length = dataloader_iter.iterator.loaders.get_seqlen()
        else:
            seq_length = self.cfg.encoder_seq_length

        # run forward and backwards passes for an entire global batch
        # we do this inside training_step to support pipeline parallelism
        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            data_iterator=dataloader_iter,
            model=[self.model],
            num_microbatches=get_num_microbatches(),
            forward_only=False,
            seq_length=seq_length,
            micro_batch_size=self.cfg.micro_batch_size,
        )

        if losses_reduced_per_micro_batch:
            loss_tensors_list = [loss_reduced['loss'] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.vstack(loss_tensors_list)
            loss_mean = loss_tensor.mean(axis=0)
        else:
            if self.cfg.bert_binary_head == True:
                loss_mean = torch.tensor([0.0, 0.0, 0.0]).cuda()
            else:
                loss_mean = torch.tensor([0.0, 0.0]).cuda()

        # when using sequence parallelism, the sequence parallel layernorm grads must be all-reduced
        if self.cfg.get('tensor_model_parallel_size', 1) > 1 and self.cfg.get('sequence_parallel', False):
            self.allreduce_sequence_parallel_gradients()

        if self.with_distributed_adam:
            # synchronize asynchronous grad reductions
            # note: not necessary, but reduces performance degradation
            # from multiple simultaneous NCCL calls
            self._optimizer._finish_bucket_grad_sync()
        elif self.megatron_amp_O2:
            if self.cfg.get('pipeline_model_parallel_size', 1) > 1 or self.cfg.get('sequence_parallel', False):
                # when using pipeline parallelism grads must be all-reduced after the pipeline (not asynchronously)
                self._optimizer.allreduce_main_grads()
        else:
            # async grad allreduce is not currently implemented for O1/autocasting mixed precision training
            # so we all-reduce gradients after the pipeline
            self.allreduce_gradients()  # @sangkug we think this is causing memory to blow up (hurts perf)

        if self.cfg.get('pipeline_model_parallel_size', 1) > 1:
            # when using pipeline parallelism the first and last stage must keep embeddings in sync
            self.allreduce_first_last_embeddings()

        torch.distributed.broadcast(loss_mean, get_last_rank())

        if self.torch_dtype == torch.float16:
            loss_scale = self.trainer.precision_plugin.scaler._scale
            if loss_scale is not None:
                self.log('loss_scale', loss_scale, batch_size=1)
        
        if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
            # Reduced loss for logging.
            self.log('reduced_train_loss', loss_mean[0], prog_bar=True, batch_size=1)
            if len(loss_mean) > 2:
                self.log('reduced_lm_train_loss', loss_mean[1], prog_bar=True, batch_size=1)
                self.log('reduced_sop_train_loss', loss_mean[2], prog_bar=True, batch_size=1)
            lr = self._optimizer.param_groups[0]['lr']
            self.log('lr', lr, batch_size=1)
            self.log('global_step', self.trainer.global_step, prog_bar=True, batch_size=1)
            self.log(
                'consumed_samples', self._compute_consumed_samples_after_training_step(), prog_bar=True, batch_size=1,
            )
        return loss_mean[0]
    
    def validation_step(self, dataloader_iter, batch_idx):
        self.build_train_valid_test_datasets()
        # self.setup_training_data(self.cfg.data)
        self.setup_validation_data(self.cfg.data)

        # Check if iterator is exhausted
        self.in_valid = True
        dataloader_iter, done = self._val_iterator_done(dataloader_iter)

        if done:
            return
        prefix = "test" if self.trainer.testing else "val"
        if self.cfg.data.dataloader_type == "LDDL":
            seq_length = dataloader_iter.iterator.get_seqlen()
        else:
            seq_length = self.cfg.encoder_seq_length

        fwd_bwd_function = get_forward_backward_func()

        losses_reduced_per_micro_batch = fwd_bwd_function(
            forward_step_func=self.get_forward_output_and_loss_func(),
            data_iterator=dataloader_iter,
            model=[self.model],
            num_microbatches=get_num_microbatches(),
            forward_only=True,
            seq_length=seq_length,
            micro_batch_size=self.cfg.micro_batch_size,
        )
        self.in_valid = False

        if losses_reduced_per_micro_batch:
            loss_tensors_list = [loss_reduced['loss'] for loss_reduced in losses_reduced_per_micro_batch]
            loss_tensor = torch.vstack(loss_tensors_list)
            loss_mean = loss_tensor.mean(axis=0)
        else:
            loss_mean = torch.tensor([0.0]).cuda()

        loss = loss_mean[0]
        self.validation_step_outputs.append(loss) if prefix == 'val' else self.test_step_outputs.append(loss)
        return loss
    
    def get_forward_output_and_loss_func(self):
        def fwd_output_and_loss_func(dataloader_iter, model, checkpoint_activations_all_layers=None):
  
            if parallel_state.get_pipeline_model_parallel_world_size() == 1:
                batches = next(
                    dataloader_iter
                ) 
                if self.in_valid:
                    
                    for ccc, ddd in enumerate(self._validation_dl):
                        if ccc == self.valid_counter:
                            batches = ddd
                    self.valid_counter += 1
                # for each element in batches, there should be 1 anchor, 1 positive, and n negatives
                # In Bert dataset (like Pile), every batch has tokens, types, sentence_order, loss_mask, lm_labels, padding_mask
                # For Sbert, we want the batch to be a list of [anchors, positives, negatives1, negatives2, ..., ] so that every of the anchors/positives/negatives are the same as the batch in pile dataset
                # batches = [anchors, positives, negatives1, negatives2]
                (
                    tokens_batch,
                    types_batch,
                    sentence_order_batch,
                    loss_mask_batch,
                    lm_labels_batch,
                    padding_mask_batch,
                ) = ([], [], [], [], [], [])
                for batch in batches:
                    tokens, types, sentence_order, loss_mask, lm_labels, padding_mask = (
                        batch['input_ids'].cuda(non_blocking=True),
                        batch['token_type_ids'].cuda(non_blocking=True),
                        None,  # batch['is_random'].cuda(non_blocking=True),
                        None,  # batch['loss_mask'].cuda(non_blocking=True),
                        None,  # batch['labels'].cuda(non_blocking=True),
                        batch['attention_mask'].cuda(non_blocking=True),
                    )
                    tokens_batch.append(tokens)
                    types_batch.append(types)
                    sentence_order_batch.append(sentence_order)
                    loss_mask_batch.append(loss_mask)
                    lm_labels_batch.append(lm_labels)
                    padding_mask_batch.append(padding_mask)
            else:
                batch = next(dataloader_iter)
                if parallel_state.is_pipeline_first_stage():
                    tokens = batch['text'].cuda(non_blocking=True)
                    types = batch['types'].cuda(non_blocking=True)
                    sentence_order = batch['is_random'].cuda(non_blocking=True)
                    padding_mask = batch['padding_mask'].cuda(non_blocking=True)
                    loss_mask, lm_labels = None, None
                elif parallel_state.is_pipeline_last_stage():
                    loss_mask = batch['loss_mask'].cuda(non_blocking=True)
                    lm_labels = batch['labels'].cuda(non_blocking=True)
                    sentence_order = batch['is_random'].cuda(non_blocking=True)
                    padding_mask = batch['padding_mask'].cuda(non_blocking=True)
                    tokens, types = None, None
                else:
                    padding_mask = batch['padding_mask'].cuda(non_blocking=True)
                    sentence_order = batch['is_random'].cuda(non_blocking=True)
                    tokens, types, loss_mask, lm_labels = None, None, None, None

            if not self.cfg.bert_binary_head:
                types = None

            forward_args = [
                {"input_ids": tokens, "token_type_ids": types, "attention_mask": padding_mask}
                for tokens, padding_mask, types in zip(
                    tokens_batch, padding_mask_batch, types_batch
                )
            ]
            if self.in_valid:
                # print(f"forward_args = {forward_args}")
                from torch import tensor
    #             forward_args = [{'input_ids': tensor([[  101, 23032,  1024,  2106,  1996,  8136, 27826,  2208,  2272,  2041,
    #       2077,  1996,  3185,   102]], device='cuda:0'), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0')}, {'input_ids': tensor([[  101,  6019,  1024,  8136, 27826,  1010,  2036,  2124,  2004, 13679,
    #      28983,  1024,  8136, 27826,  2090,  2541,  1998,  2289,  1010,  2003,
    #       1037,  2865,  6329,  2008,  7940,  2007,  2019,  2895,  1011,  6172,
    #       2678,  2208,  2186,  2580,  2011,  2329, 10355,  2194,  4563,  2640,
    #       1012,  3839,  3079,  2011,  1041, 13820,  2015,  9123,  1010,  2059,
    #       2011,  2675,  4372,  7646,  2044,  2037,  7654,  1997,  1041, 13820,
    #       2015,  1999,  2268,  1010,  1996,  6329,  7679,  2006,  1037,  7214,
    #       2329, 18821, 13679, 28983,  1010,  2040,  7930,  2105,  1996,  2088,
    #       6575,  2005,  2439, 25762,  1998,  1999,  8873,  7096, 15172,  4795,
    #      16623,  1998,  8435,  1012,  1996, 11247,  3227,  7679,  2105,  2895,
    #       1011,  6172,  8993,  1997, 10058,  1010, 13729, 19672,  1010,  6583,
    #       5737, 16961, 10420, 10058,  3561,  2007, 16735,  1010,  1998,  3554,
    #       3365,  6716,  1012,  3176,  2865,  2038,  4961,  2039,  2105,  1996,
    #       4323,  1999,  1996,  2433,  1997,  2143, 17241,  1010,  5888,  1998,
    #       6002,  1012,   102]], device='cuda:0'), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    #    device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
    #    device='cuda:0')}, {'input_ids': tensor([[  101,  6019,  1024,  4125,  1997,  1996,  8136, 27826,  2001,  2207,
    #       2006,  2184,  2281,  2325,  1010,  1998,  1996,  3645,  2544,  2001,
    #       2207,  2006,  2654,  2254,  2355,  1012,  7513,  4835,  2001,  1996,
    #       2208,  1005,  1055,  6674,  2005, 12202,  9475,  1998, 12202,  2028,
    #       1012,  2019,  2324,  1011,  3277,  5021,  2186,  1010,  8136, 27826,
    #       1010,  2211,  4772,  1999,  2220,  2297,  1012,  2550,  2011,  2601,
    #       3586,  5888,  1998,  2517,  2011, 10975,  4017, 20318,  2102,  1998,
    #      18576, 14072,  1010,  1996,  5888,  2958,  2094,  1996,  6578,  2090,
    #       1996,  2286,  2128, 27927,  1998,  4125,  1997,  1996,  8136, 27826,
    #       1998,  4541,  1996,  6438,  1997,  2070,  3905,  3494,  1999,  1996,
    #       8297,  1012,  7513,  2207,  1037,  4125,  1997,  1996,  8136, 27826,
    #      12202,  2028, 14012,  1010,  2164,  2019, 12202,  2028, 10122,  1010,
    #       1037,  3642,  2005,  8136, 27826,  1024, 15764,  3179,  1998,  1996,
    #       2208,  1012,  1037, 10018,  1005,  1055,  3179,  2443,  1037,  2260,
    #       1011,  4960,  6231,  1997, 13679,  1010,  1037,  3886,  8654,  1010,
    #       1037, 12323, 13016,  1998,  1037, 15059,  1997, 13679,  1005,  1055,
    #       3485,  1012,  1037,  2161,  3413,  2443,  1996,  2918,  2208,  1010,
    #       3176, 22054,  1010,  4255,  1998,  5590,  5329,  1010,  1998,  3229,
    #       2000, 26720,  4180,  1012,  2399, 14399,  3653,  8551,  2545,  2018,
    #       7262,  3229,  2000,  1996,  4151,  2543,  4003,  5308,  1010,  2029,
    #       2064,  2022,  2109,  1999,  1996,  2208,  1005,  1055, 15014,  5549,
    #       1012,  7513,  2444,  1011, 18498,  1037,  7691,  4908,  5821,  2724,
    #       2006, 19435,  1012,  2694,  1012,  2809, 10584,  3061,  1999,  2392,
    #       1997,  1037,  2148, 24298,  2395,  4908,  2020, 13532,  2000,  2367,
    #       8401,  4633,  3785,  1010,  2029,  2020,  5444,  2011, 19435,  1005,
    #       1055,  7193,  1012,  1996, 10832, 16762,  1996,  4633,  6493,  2363,
    #       1037,  1036,  1036,  8136, 27826,  1011, 11773,  4440,  1005,  1005,
    #       2867,  2071,  7796,  1999,  1011,  2208, 19054,  2011,  8019,  1006,
    #       1998, 21935,  2007,  1007, 19435,  2444,  1011, 11058,  1999,  5590,
    #       5549,  1012,   102]], device='cuda:0'), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1]], device='cuda:0')}, {'input_ids': tensor([[  101,  6019,  1024,  8136, 27826,  1024,  5315,  2003,  1037,  2289,
    #       2895,  1011,  6172,  2678,  2208,  1010,  2112,  1997,  1996,  8136,
    #      27826,  2186,  1012,  2009,  2003,  1037, 12661,  1013,  2128,  1011,
    #      16603,  1997,  1996,  2034,  2678,  2208,  1999,  1996,  2186,  1010,
    #       1996,  2434,  2727,  8136, 27826,  1012,  2009,  3594,  2019,  5301,
    #       2544,  1997,  1996,  5722,  2208,  3194,  1010,  1998,  2009,  2950,
    #       2035,  1997,  1996,  2434, 10058,  2013,  8136, 27826,  1012,   102]],
    #    device='cuda:0'), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    #    device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
    #    device='cuda:0')}, {'input_ids': tensor([[  101,  6019,  1024,  1996,  2208,  2001,  2623,  2000,  2022,  2405,
    #       2011, 27681,  4091,  2399,  2005,  2195, 22659,  2823,  2105,  1999,
    #       2760,  2164,  9160,  1018,  1010, 12202,  2028,  1010,  2047, 10022,
    #       7605,  2015,  1010,  1998, 10022,  6942,  1012,  1010,   102]],
    #    device='cuda:0'), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0')}, {'input_ids': tensor([[  101,  6019,  1024,  1996,  2143,  2441,  2006,  2281,  1019,  1010,
    #       2432,  1010,  2004, 14255, 18684,  2099,  1005,  1055,  2034,  2143,
    #       2000,  2022,  6758, 18720,  1006,  2005,  1036,  1036,  2895,  4808,
    #       1005,  1005,  1007,  1012,  2049,  8900,  2713,  2001,  5642,  2007,
    #       1037, 14255, 18684,  2099,  2460,  2143,  5391,  2378,  1005,  1012,
    #       1996, 10319,  3049,  2443,  2019,  2880,  4037,  2007,  2678,  9214,
    #       1010,  2399,  1010,  1998,  6140,  3085, 28663,  1012,  2096, 14255,
    #      18684,  2099,  6334,  2178, 10911,  2007,  1996,  9788,  2015,  1010,
    #       3889,  5841,  2001,  7861, 12618, 18450,  1999,  1037,  2270, 13552,
    #       2007,  1996,  2132,  1997,  2049,  4353,  4256,  1010,  1996, 10598,
    #       6373,  2194,  1012,  2023,  2052,  2776,  2599,  2000,  1996, 15068,
    #      16643,  3070,  1997,  2745,  1041,  2483,  3678,  1998,  6373,  1005,
    #       1055,  7654,  1997, 14255, 18684,  2099,  1996,  2206,  2095,  1012,
    #        102]], device='cuda:0'), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0')}]
            

            ''' if not self.mcore_bert:
                forward_args["checkpoint_activations_all_layers"] = checkpoint_activations_all_layers
                forward_args["model"] = model
                forward_args["token_type_ids"] = types
            else:
                forward_args["tokentype_ids"] = types'''
            # self.model.eval()
            output_tensor = None
            if self.mcore_bert:
                output_tensor = model(**forward_args)
            else:
                output_tensor = [self.forward(**forward_arg).permute(1,0) for forward_arg in forward_args]
            # if self.stop_criteria:
            #     print(f"output_tensor = {output_tensor}")
 
            def loss_func(output_tensor):

                loss_dict = self.loss_func(output_tensor)

                if 'sop loss' in loss_dict:
                    lm_loss = loss_dict['lm loss']
                    sop_loss = loss_dict['sop loss']
                    loss = lm_loss + sop_loss
                    reduced_loss = average_losses_across_data_parallel_group([loss, lm_loss, sop_loss])
                else:
                    lm_loss = loss_dict['lm loss']
                    loss = lm_loss
                    reduced_loss = average_losses_across_data_parallel_group([loss, lm_loss])

                return loss, {'loss': reduced_loss}

            return output_tensor, loss_func

        return fwd_output_and_loss_func

    def loss_func(self, output_tensor):
        queries = output_tensor[0]
        positives = output_tensor[1]

        pos_inbatch_negs_scores = torch.mm(queries, positives.transpose(0, 1))

        hard_negs = output_tensor[2:]

        hard_negs_scores = (
            torch.multiply(queries.unsqueeze(0).repeat(len(hard_negs), 1, 1), torch.stack(hard_negs),).sum(axis=-1).T
        )

        scores = torch.cat([pos_inbatch_negs_scores, hard_negs_scores], axis=1)

        scores *= self.scale

        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )  # Indices of the (query, positive) pairs
        if self.stop_criteria and self.in_valid:

            print(f"val loss = {self.cross_entropy_loss(scores, labels)}")

        return {'lm loss': self.cross_entropy_loss(scores, labels)}

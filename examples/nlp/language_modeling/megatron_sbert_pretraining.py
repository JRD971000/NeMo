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

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict

from nemo.collections.nlp.models.language_modeling.megatron_sbert_model import MegatronSBertModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronBertTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
import torch

def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

@hydra_runner(config_path="conf", config_name="megatron_bert_config")
def main(cfg) -> None:
    if cfg.model.data.dataloader_type != "LDDL":
        mp.set_start_method("spawn", force=True)

    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    trainer = MegatronBertTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    model_cfg = MegatronSBertModel.restore_from(
        restore_path=cfg.restore_from_path,
        trainer=trainer,
        save_restore_connector=NLPSaveRestoreConnector(),
        return_config=True,
    )
    model_cfg.data.data_prefix = cfg.model.data.data_prefix
    model_cfg.micro_batch_size = cfg.model.micro_batch_size
    model_cfg.global_batch_size = cfg.model.global_batch_size
    model_cfg.optim.lr = cfg.model.optim.lr
    model_cfg.optim.sched.min_lr = cfg.model.optim.sched.min_lr
    model_cfg.data.dataloader_type = "single"
    model_cfg.optim.sched.warmup_steps = cfg.model.optim.sched.warmup_steps
    model_cfg.encoder_seq_length = cfg.model.encoder_seq_length
    model_cfg.tokenizer.library = cfg.model.tokenizer.library
    model_cfg.tokenizer.type = cfg.model.tokenizer.type
    model_cfg.data.data_prefix = cfg.model.data.data_prefix
    model_cfg.tokenizer.do_lower_case = cfg.model.tokenizer.do_lower_case
    model_cfg.data.evaluation_sample_size = cfg.model.data.evaluation_sample_size
    model_cfg.data.hard_negatives_to_train = cfg.model.data.hard_negatives_to_train
    model_cfg.data.evaluation_steps = cfg.model.data.evaluation_steps
    model_cfg.optim.weight_decay = 0.01
    assert (
        model_cfg.micro_batch_size * cfg.trainer.devices == model_cfg.global_batch_size
    ), "Gradiant accumulation is not supported for contrastive learning yet"

    OmegaConf.set_struct(model_cfg, True)
    with open_dict(model_cfg):
        model_cfg.precision = trainer.precision

    model = MegatronSBertModel.restore_from(
        restore_path=cfg.restore_from_path,
        trainer=trainer,
        save_restore_connector=NLPSaveRestoreConnector(),
        override_config_path=model_cfg,
        strict=True,
    )

    # model = MegatronBertModel(cfg.model, trainer)

    trainer.fit(model)


if __name__ == '__main__':
    main()

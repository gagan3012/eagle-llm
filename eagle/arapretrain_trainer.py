from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments
from trl import SFTTrainer
import torch

major_version, minor_version = torch.cuda.get_device_capability()
SUPPORTS_BFLOAT16 = (major_version >= 8)
def is_bf16_supported(): return SUPPORTS_BFLOAT16
torch.cuda.is_bf16_supported = is_bf16_supported

@dataclass
class AraPretrainTrainingArguments(TrainingArguments):
    embedding_learning_rate : Optional[float] = field(
        default = None,
        metadata = {"help" : "Different learning rates for embeddings and lm_head."}
    )
pass


def _create_AraPretrain_optimizer(
    model,
    optimizer_cls,
    optimizer_kwargs,
    embedding_lr = 1e-5,
):
    lr = optimizer_kwargs["lr"]
    weight_decay = optimizer_kwargs.get("weight_decay", 0.0)

    param_groups = \
    {
        "non_embeddings" : {},
        "embeddings"     : {},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if name.endswith("modules_to_save.default.weight"):
            partial_name = name[:-len(".modules_to_save.default.weight")]
            partial_name = partial_name[partial_name.rfind(".")+1:]
            print(f"AraPretrain: Setting lr = {embedding_lr:.2e} instead of {lr:.2e} for {partial_name}.")
            param_groups["embeddings"]    [name] = param
        else:
            param_groups["non_embeddings"][name] = param
        pass
    pass

    optimizer_grouped_parameters = [
        {
            "params"       : list(param_groups["non_embeddings"].values()),
            "weight_decay" : weight_decay,
            "lr"           : lr,
        },
        {
            "params"       : list(param_groups["embeddings"].values()),
            "weight_decay" : weight_decay,
            "lr"           : embedding_lr,
        },
    ]
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer
pass


class AraPretrainTrainer(SFTTrainer):
    def create_optimizer(self):
        embedding_learning_rate = getattr(self.args, "embedding_learning_rate", None)
        if embedding_learning_rate is None: return super().create_optimizer()

        if self.optimizer is None:
            optimizer_cls, optimizer_kwargs = SFTTrainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = _create_AraPretrain_optimizer(
                self.model,
                optimizer_cls,
                optimizer_kwargs,
                embedding_learning_rate,
            )
        pass
        return self.optimizer
    pass
pass
from typing import Any, Dict, List, Sequence, Tuple, Union

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from torch.optim import Optimizer
from torchvision.models import ResNet
from torchvision.models.resnet import BasicBlock

from src.common.utils import iterate_elements_in_batches, render_images


class MnistResNet(ResNet):
    # https://zablo.net/blog/post/using-resnet-for-mnist-in-pytorch-tutorial/
    def __init__(self) -> None:
        super(MnistResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )


class MyModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        self.resnet = MnistResNet()

        metric = pl.metrics.Accuracy()
        self.train_accuracy = metric.clone()
        self.val_accuracy = metric.clone()
        self.test_accuracy = metric.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

    def step(self, x, y) -> Dict[str, torch.Tensor]:
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return {"logits": logits, "loss": loss}

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        out = self.step(x, y)
        self.train_accuracy(torch.softmax(out["logits"], dim=-1), y)
        self.log_dict(
            {
                "train_acc": self.train_accuracy,
                "train_loss": out["loss"],
            },
            on_epoch=True,
        )
        return out["loss"]

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        out = self.step(x, y)
        self.val_accuracy(torch.softmax(out["logits"], dim=-1), y)
        self.log_dict(
            {
                "val_acc": self.val_accuracy,
                "val_loss": out["loss"],
            },
        )
        return {
            "image": x,
            "y_true": y,
            "logits": out["logits"],
            "val_loss": out["loss"],
        }

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y = batch
        out = self.step(x, y)
        self.test_accuracy(torch.softmax(out["logits"], dim=-1), y)
        self.log_dict(
            {
                "test_acc": self.test_accuracy,
                "test_loss": out["loss"],
            },
        )
        return {
            "image": x,
            "y_true": y,
            "logits": out["logits"],
            "val_loss": out["loss"],
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        batch_size = self.cfg.data.datamodule.batch_size.val
        images = []
        for output_element in iterate_elements_in_batches(
            outputs, batch_size, self.cfg.logging.n_elements_to_log
        ):
            rendered_image = render_images(output_element["image"], autoshow=False)
            caption = f"y_pred: {output_element['logits'].argmax()}  [gt: {output_element['y_true']}]"
            images.append(
                wandb.Image(
                    rendered_image,
                    caption=caption,
                )
            )
        self.logger.experiment.log({"Validation Images": images}, step=self.global_step)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        batch_size = self.cfg.data.datamodule.batch_size.test
        images = []
        for output_element in iterate_elements_in_batches(
            outputs, batch_size, self.cfg.logging.n_elements_to_log
        ):
            rendered_image = render_images(output_element["image"], autoshow=False)
            caption = f"y_pred: {output_element['logits'].argmax()}  [gt: {output_element['y_true']}]"
            images.append(
                wandb.Image(
                    rendered_image,
                    caption=caption,
                )
            )
        self.logger.experiment.log({"Test Images": images}, step=self.global_step)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(
            self.cfg.optim.optimizer, params=self.parameters()
        )
        scheduler = hydra.utils.instantiate(self.cfg.optim.lr_scheduler, optimizer=opt)
        return [opt], [scheduler]

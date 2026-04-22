# Copyright 2024 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import pickle
from typing import Any, Dict, List, Sequence, Tuple

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.nnclr import nnclr_loss_func
from solo.methods.base import BaseMomentumMethod
from solo.utils.misc import gather, omegaconf_select
from solo.utils.momentum import initialize_momentum_params
from solo.utils.positional_encodings import PositionalEncodingPermute1D, Summer


class All4One(BaseMomentumMethod):
    queue: torch.Tensor

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)

        self.temperature: float = cfg.method_kwargs.temperature
        self.queue_size: int = cfg.method_kwargs.queue_size
        self.losses_names = ["att_nnclr_loss", "nnclr_loss", "feature_loss"]
        self.losses_weights = cfg.method_kwargs.get("losses_weights", {"att_nnclr_loss": 0.5, "nnclr_loss": 0.5, "feature_loss": 5})
        assert set(self.losses_names) == set(self.losses_weights)

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        pred_hidden_dim: int = cfg.method_kwargs.pred_hidden_dim

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
            nn.BatchNorm1d(proj_output_dim),
        )

        # momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
            nn.BatchNorm1d(proj_output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )

        # second predictor
        self.predictor2 = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )

        # internal transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_output_dim,
            nhead=8,
            dim_feedforward=proj_output_dim * 2,
            batch_first=True,
            dropout=0.1,
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # positional encoder
        self.pos_enc = Summer(PositionalEncodingPermute1D(5))

        # queue
        self.register_buffer("queue", torch.randn(self.queue_size, proj_output_dim))
        self.register_buffer("queue_y", -torch.ones(self.queue_size, dtype=torch.long))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # NN index queue
        self.register_buffer("queue_index", -torch.ones(self.queue_size, dtype=torch.long))

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(All4One, All4One).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.temperature")

        cfg.method_kwargs.queue_size = omegaconf_select(cfg, "method_kwargs.queue_size", 65536)

        return cfg

    @property
    def extra_learnable_params(self) -> List[dict]:
        extra_learnable_params = [
            {"params": self.projector.parameters()},
            {"params": self.predictor.parameters()},
            {"params": self.predictor2.parameters()},
            {"params": self.transformer_encoder.parameters(), "lr": 0.1},
        ]
        return extra_learnable_params

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """
        return super().learnable_params + self.extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs

    @torch.no_grad()
    def dequeue_and_enqueue(self, z: torch.Tensor, y: torch.Tensor, idx: torch.Tensor):
        """Adds new samples and removes old samples from the queue in a fifo manner. Also stores
        the labels of the samples.

        Args:
            z (torch.Tensor): batch of projected features.
            y (torch.Tensor): labels of the samples in the batch.
            idx (torch.Tensor): batch of indexes
        """

        z = gather(z)
        y = gather(y)
        idx = gather(idx)

        batch_size = z.shape[0]

        ptr = int(self.queue_ptr)  # type: ignore
        assert self.queue_size % batch_size == 0

        self.queue[ptr : ptr + batch_size, :] = z
        self.queue_y[ptr : ptr + batch_size] = y  # type: ignore

        # NN indexes
        self.queue_index[ptr : ptr + batch_size] = idx

        ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr  # type: ignore

    @torch.no_grad()
    def find_nn(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Finds the nearest neighbors of a sample.

        Args:
            z (torch.Tensor): a batch of projected features.

        Returns:
            torch.Tensor:
                indexes of the first NNs.
            torch.Tensor:
                extracted batch of NNs.
            torch.Tensor:
                NN indexes.
            torch.Tensor:
                NN labels.
        """

        idxx = (z @ self.queue.T).max(dim=1)[1]

        _, idx = (z @ self.queue.T).topk(5, dim=1)

        nn = self.queue[idx]
        nn_idx = self.queue_index[idx]
        nn_lb = self.queue_y[idx]

        return idxx, nn, nn_idx, nn_lb

    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict:
        return super().momentum_forward(X)

    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs forward pass of the online backbone, projector and predictor.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]:
                a dict containing the outputs of the parent, the projected features and the
                predicted features.
        """

        out = super().forward(X, *args, **kwargs)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        return {**out, "z": z, "p": p}

    def off_diagonal(self, x):
        """Extracts off-diagonal elements.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            torch.Tensor:
                flattened off-diagonal elements.
        """
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def save_NN(self, img_indexes, nn1_idx, nn1_lb):
        """Auxiliar function to store the NNs.

        Args:
            img_indexes (torch.Tensor): batch of image indexes in tensor format.
            nn1_idx (torch.Tensor): batch of NN indexes in tensor format.
            nn1_lb (torch.Tensor): batch of NN labels in tensor format.

        """

        with open(f"NNIDX/FirstNN/{self.current_epoch}__{self.global_step}__NNS.pickle", "wb") as f:
            pickle.dump(nn1_idx.cpu().numpy(), f)

        with open(f"NNIDX/FirstNN/{self.current_epoch}__{self.global_step}__IDX.pickle", "wb") as f:
            pickle.dump(img_indexes.cpu().numpy(), f)

        with open(
            f"NNIDX/FirstNN/{self.current_epoch}__{self.global_step}__Labels.pickle", "wb"
        ) as f:
            pickle.dump(nn1_lb.cpu().numpy(), f)

    def forward_embeddings(self, batch: Sequence[Any], batch_idx: int) -> Dict[str, Any]:
        """Apply backbone: batch → backbone embeddings dict.

        Returns:
            Dict with keys: targets, feats1/2, momentum_feats1/2.
        """
        out = super().training_step(batch, batch_idx)
        feats1, feats2 = out["feats"]
        momentum_feats1, momentum_feats2 = out["momentum_feats"]

        return {
            "targets": batch[-1],
            "img_indexes": batch[0],
            "feats1": feats1,
            "feats2": feats2,
            "momentum_feats1": momentum_feats1,
            "momentum_feats2": momentum_feats2,
        }

    # --- intermediate representation builders ---

    @torch.no_grad()
    def _momentum_projections(
        self, momentum_feats1: torch.Tensor, momentum_feats2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        momentum_z1 = F.normalize(self.momentum_projector(momentum_feats1), dim=-1)
        momentum_z2 = F.normalize(self.momentum_projector(momentum_feats2), dim=-1)
        return momentum_z1, momentum_z2

    def _online_projections(
        self, feats1: torch.Tensor, feats2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.projector(feats1), self.projector(feats2)

    def _predictions(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.predictor(z1), self.predictor(z2), self.predictor2(z1), self.predictor2(z2)

    def _nearest_neighbors(
        self, momentum_z1: torch.Tensor, momentum_z2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx1, nn1, *_ = self.find_nn(momentum_z1)
        _, nn2, _, _ = self.find_nn(momentum_z2)
        return idx1, nn1, nn2

    def _transformer_encodings(
        self,
        nn1: torch.Tensor,
        nn2: torch.Tensor,
        p1_2: torch.Tensor,
        p2_2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rich_emb1 = self.transformer_encoder(self.pos_enc(nn1))[:, 0, :]
        rich_emb2 = self.transformer_encoder(self.pos_enc(nn2))[:, 0, :]
        strange_emb1 = self.transformer_encoder(
            self.pos_enc(torch.cat((p1_2.unsqueeze(1), nn1), 1)[:, :5, :])
        )[:, 0, :]
        strange_emb2 = self.transformer_encoder(
            self.pos_enc(torch.cat((p2_2.unsqueeze(1), nn2), 1)[:, :5, :])
        )[:, 0, :]
        return rich_emb1, rich_emb2, strange_emb1, strange_emb2

    # --- individual losses ---

    def _att_nnclr_loss(
        self,
        rich_emb1: torch.Tensor,
        rich_emb2: torch.Tensor,
        strange_emb1: torch.Tensor,
        strange_emb2: torch.Tensor,
    ) -> torch.Tensor:
        return (
            nnclr_loss_func(rich_emb1, strange_emb2) / 2
            + nnclr_loss_func(rich_emb2, strange_emb1) / 2
        )

    def _nnclr_loss(
        self,
        nn1: torch.Tensor,
        nn2: torch.Tensor,
        p1: torch.Tensor,
        p2: torch.Tensor,
    ) -> torch.Tensor:
        return (
            nnclr_loss_func(nn1[:, 0, :], p2, temperature=self.temperature) / 2
            + nnclr_loss_func(nn2[:, 0, :], p1, temperature=self.temperature) / 2
        )

    def _feature_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        momentum_z1: torch.Tensor,
        momentum_z2: torch.Tensor,
    ) -> torch.Tensor:
        p1_n = F.normalize(momentum_z1, dim=0)
        p2_n = F.normalize(momentum_z2, dim=0)
        z1_n = F.normalize(z1, dim=0)
        z2_n = F.normalize(z2, dim=0)

        c1 = p1_n.T @ z2_n
        c2 = p2_n.T @ z1_n

        on_diag = (
            (torch.diagonal(c1).add(-1).pow(2).mean() + torch.diagonal(c2).add(-1).pow(2).mean())
            * 0.5
        ).sqrt()
        off_diag = (
            (self.off_diagonal(c1).pow(2).mean() + self.off_diagonal(c2).pow(2).mean()) * 0.5
        ).sqrt()
        return 0.5 * (on_diag + off_diag)

    def _class_loss(
        self, feats1: torch.Tensor, feats2: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return (
            F.cross_entropy(self.classifier(feats1.detach()), targets, ignore_index=-1)
            + F.cross_entropy(self.classifier(feats2.detach()), targets, ignore_index=-1)
        ) / 2

    # --- orchestrator ---

    def compute_losses(self, embs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Apply All4One modules and compute all losses from backbone embeddings.

        Args:
            embs: output of forward_embeddings.

        Returns:
            Dict with keys: class_loss, att_nnclr_loss, nnclr_loss,
            feature_loss, z1, z2, idx1.
        """
        feats1, feats2 = embs["feats1"], embs["feats2"]

        momentum_z1, momentum_z2 = self._momentum_projections(
            embs["momentum_feats1"], embs["momentum_feats2"]
        )
        z1, z2 = self._online_projections(feats1, feats2)
        p1, p2, p1_2, p2_2 = self._predictions(z1, z2)
        idx1, nn1, nn2 = self._nearest_neighbors(momentum_z1, momentum_z2)
        rich_emb1, rich_emb2, strange_emb1, strange_emb2 = self._transformer_encodings(
            nn1, nn2, p1_2, p2_2
        )

        feature_loss = self._feature_loss(z1, z2, momentum_z1, momentum_z2)

        self.dequeue_and_enqueue(momentum_z1, embs["targets"], embs["img_indexes"])

        return {
            "class_loss": self._class_loss(feats1, feats2, embs["targets"]),
            "att_nnclr_loss": self._att_nnclr_loss(rich_emb1, rich_emb2, strange_emb1, strange_emb2),
            "nnclr_loss": self._nnclr_loss(nn1, nn2, p1, p2),
            "feature_loss": feature_loss,
            "z1": z1,
            "z2": z2,
            "idx1": idx1,
        }

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for All4One reusing BaseMomentumMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of All4One and classification loss.
        """
        targets = batch[-1]

        embs = self.forward_embeddings(batch, batch_idx)
        losses = self.compute_losses(embs)

        ssl_loss = sum(self.losses_weights[name] * losses[name] for name in self.losses_names)

        nn_acc = (targets == self.queue_y[losses["idx1"]]).sum() / targets.size(0)
        z_std = (
            F.normalize(losses["z1"], dim=-1).std(dim=0).mean()
            + F.normalize(losses["z2"], dim=-1).std(dim=0).mean()
        ) / 2

        self.log_dict(
            {
                "train_nnclr_loss": losses["nnclr_loss"],
                "train_att_nnclr_loss": losses["att_nnclr_loss"],
                "train_feature_loss": losses["feature_loss"],
                "train_nn_acc": nn_acc,
                "train_z_std": z_std,
                "train_comb_loss": ssl_loss,
            },
            on_epoch=True,
            sync_dist=True,
        )

        return ssl_loss + losses["class_loss"]

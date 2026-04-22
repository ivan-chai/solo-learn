from contextlib import contextmanager
from typing import List

import omegaconf
import torch
import torch.nn.functional as F
from aligned_hpo import AlignedHPOptimizer, HPO_STAGE_DOWNSTREAM
from solo.methods.all4one import All4One


class HPOAll4One(All4One):
    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)

        self._OPTIMIZERS = dict(self._OPTIMIZERS)
        self.base_optimizer = self.optimizer

        self.automatic_optimization = False
        self.hpo_losses = list(sorted(self.losses_names))
        self.downstream_loss = "class_loss"

        hpo_kwargs = dict(cfg.hpo_kwargs)
        self.hpo_params = hpo_kwargs.pop("hpo_params", None)
        self.hp_group_params = hpo_kwargs.pop("hp_group_params", None)
        self.cache_embedding_gradients = hpo_kwargs.pop("cache_embedding_gradients", False)
        self.gradient_clip_val = hpo_kwargs.pop("gradient_clip_val", None)

        initial_weights = hpo_kwargs.pop("initial_weights", None)
        if initial_weights is not None:
            initial_weights = torch.tensor(initial_weights, dtype=torch.get_default_dtype())
        else:
            initial_weights = torch.ones([len(self.hpo_losses)])
        if initial_weights.shape != (len(self.hpo_losses),):
            raise ValueError(f"Initial weights shape mismatch: {initial_weights.shape} != ({len(self.hpo_losses)})")
        self.loss_weights = torch.nn.Parameter(initial_weights)
        assert not hpo_kwargs, set(hpo_kwargs)

    def compress_embeddings(self, embeddings):
        names = ["feats1", "feats2"]
        embeddings = dict(embeddings)
        compressed = torch.stack([embeddings.pop(name) for name in names], 0)  # (2, B, D).
        assert compressed.ndim == 3
        meta = embeddings  # Remaining keys.
        meta["batch_size"] = compressed.shape[1]
        return compressed.flatten(), meta

    def decompress_embeddings(self, embeddings, meta):
        names = ["feats1", "feats2"]
        meta = dict(meta)
        batch_size = meta.pop("batch_size")
        embeddings = embeddings.reshape(len(names), batch_size, -1)
        embeddings = {name: embeddings[i] for i, name in enumerate(names)}
        embeddings.update(meta)
        return embeddings

    def training_step(self, batch, batch_idx):
        # Unpack (batch, dataloader_idx) when InterleavedLoader is active.
        if isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[1], int):
            batch, dataloader_idx = batch
        else:
            dataloader_idx = 0

        opt = self.optimizers()
        if opt.use_validation:
            raise NotImplementedError("Val set is not supported")

        embeddings = self.forward_embeddings(batch, batch_idx)
        embeddings, meta = self.compress_embeddings(embeddings)

        if opt.encoder_decoder:
            # Detach embeddings.
            encoder_embeddings = embeddings
            embeddings = encoder_embeddings.detach().clone()
            embeddings.requires_grad = True

        use_cached_grads = opt.encoder_decoder and self.cache_embedding_gradients
        if use_cached_grads:
            raise NotImplementedError("Can't cache gradients.")

        losses = self.compute_losses(self.decompress_embeddings(embeddings, meta))
        metrics = {}

        def closure(down, weights, retain_graph=False, stage=None):
            opt.zero_grad()
            embeddings.grad = None
            assert len(weights) == len(self.hpo_losses)

            # Do backward pass.
            downstream_loss = losses[self.downstream_loss]
            loss = sum([w * losses[k] for k, w in zip(self.hpo_losses, weights)], down * downstream_loss)
            # DDP synchronization will be made in after_backward_hook.
            with self._no_sync():
                self.manual_backward(loss, retain_graph=retain_graph)

            if stage == HPO_STAGE_DOWNSTREAM:
                metrics["hpo_grad_norm_downstream"] = self._get_grad_norm(warn_empty_grads=False)
            elif isinstance(stage, int):
                metrics[f"hpo_grad_norm_weight_{self.hpo_losses[stage]}"] = self._get_grad_norm(warn_empty_grads=False)
            if opt.encoder_decoder:
                with torch.no_grad():
                    emb_grad_norm = torch.linalg.norm(embeddings.grad.flatten())
                if stage == HPO_STAGE_DOWNSTREAM:
                    metrics["hpo_emb_grad_norm_downstream"] = emb_grad_norm
                elif isinstance(stage, int):
                    metrics[f"hpo_emb_grad_norm_weight_{self.hpo_losses[stage]}"] = emb_grad_norm
                return embeddings

        if opt.encoder_decoder:
            def closure_encoder(z_grad):
                opt.zero_grad()
                # DDP synchronization will be made in after_backward_hook.
                encoder_embeddings.backward(z_grad.reshape(*encoder_embeddings.shape))

            def embed_fn():
                return self.compress_embeddings(self.forward_embeddings(batch, batch_idx))[0]
        else:
            closure_encoder = None
            embed_fn = None

        def after_backward_hook():
            # Synchronize gradients in DDP (mirrors _no_sync logic).
            if hasattr(self.trainer.model, "no_sync"):
                with torch.enable_grad():
                    zero_loss = 0 * sum(p.flatten()[0] for p in self.parameters())
                    self.manual_backward(zero_loss)
            if self.gradient_clip_val is not None:
                self.clip_gradients(opt, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm=self.trainer.gradient_clip_algorithm)
            self.log("grad_norm", self._get_grad_norm(), prog_bar=True)

        opt.hpo_step(closure, closure_encoder, embed_fn=embed_fn, after_backward_hook=after_backward_hook)
        hpo_grads = self.loss_weights.grad
        if hpo_grads is not None:
            hpo_grad_norm = torch.linalg.norm(hpo_grads)
            metrics["hpo_grad_norm"] = hpo_grad_norm
        metrics.update(opt.metrics)
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        sch = self.lr_schedulers()
        if sch is not None and self.scheduler_interval == "step":
            sch.step()

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector and predictor parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """
        weights_params = [dict(params=[self.loss_weights], **(self.hp_group_params or {}))]
        extra_learnable_params = self.extra_learnable_params
        base_learnable_params = super(All4One, self).learnable_params
        backbone_params = [group for group in base_learnable_params if group["name"] == "backbone"]
        base_heads_params = [group for group in base_learnable_params if group["name"] != "backbone"]
        assert backbone_params
        assert base_heads_params

        heads_params = base_heads_params + extra_learnable_params
        for group in heads_params:
            group["is_head"] = True
        return weights_params + heads_params + backbone_params

    def configure_optimizers(self):
        self.optimizer = "hpo"
        def make_optimizer(learnable_params, **kwargs):
            heads_groups = [i for i in range(len(learnable_params)) if learnable_params[i].get("is_head", False)]
            return AlignedHPOptimizer(learnable_params, self._OPTIMIZERS[self.base_optimizer],
                                      weights_names=self.hpo_losses,
                                      base_optimizer_params=kwargs,
                                      heads_groups=heads_groups,
                                      **(self.hpo_params or {}))
        self._OPTIMIZERS["hpo"] = make_optimizer
        result = super().configure_optimizers()
        return result

    def _class_loss(
        self, feats1: torch.Tensor, feats2: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        # Don't detach.
        return (
            F.cross_entropy(self.classifier(feats1), targets, ignore_index=-1)
            + F.cross_entropy(self.classifier(feats2), targets, ignore_index=-1)
        ) / 2

    @contextmanager
    def _no_sync(self):
        """Safely get no_sync regardless of strategy."""
        # self.trainer.model is the strategy-wrapped model (DDP, FSDP, etc.)
        if hasattr(self.trainer.model, "no_sync"):
            with self.trainer.model.no_sync():
                yield
        else:
            # Single GPU, CPU, or strategy that doesn't need no_sync
            yield

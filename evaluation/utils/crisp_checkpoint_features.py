"""Strict teacher checkpoint adapter; supplied CRISP forward remains inherited."""
import torch
from .crisper import Crisper
from .crisper_transformers import vit_small


class CheckpointViTS(Crisper):
    def __init__(self, checkpoint_path, output="dense", layer=-1, return_multilayer=False,
                 checkpoint_name="ibot_vits16"):
        torch.nn.Module.__init__(self)
        self.output = output
        self.return_multilayer = return_multilayer
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        teacher = checkpoint["teacher"]
        state = {}
        for name, tensor in teacher.items():
            while name.startswith(("module.", "_orig_mod.")):
                name = name.split(".", 1)[1]
            if name.startswith("backbone."):
                name = name[len("backbone."):]
            elif name.startswith(("head.", "ibot_head.")):
                continue
            state[name] = tensor
        assert state["cls_token"].shape == (1, 1, 384)
        self.vit = vit_small(patch_size=16, return_all_tokens=True)
        incompatible = self.vit.load_state_dict(state, strict=False)
        assert set(incompatible.missing_keys) <= {"norm_cls.weight", "norm_cls.bias"}, incompatible
        assert set(incompatible.unexpected_keys) <= {"masked_embed"}, incompatible
        self.vit.eval().requires_grad_(False)
        self.patch_size = 16
        self.checkpoint_name = checkpoint_name
        assert len(self.vit.blocks) == 12
        self.multilayers = [2, 5, 8, 11] if return_multilayer else [11 if layer == -1 else layer]
        dim = 384 * (2 if output == "dense-cls" else 1)
        self.feat_dim = [dim] * 4 if return_multilayer else dim
        self.layer = "-".join(str(x) for x in self.multilayers)
        print("Strict teacher loaded:", checkpoint_path, "epoch:", checkpoint.get("epoch"), flush=True)

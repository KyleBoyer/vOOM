"""Fixture-only selector for a decode-only shared-expert scheduling theory.

No serving default/profile is installed until a clean timing gate wins. The
wrapper changes no prompt, routes, output budget, or model representation.
"""
from runtime import glm5_next


class SharedOverlapProbe:
    def __init__(self, target):
        if target.cfg.model_type != "glm5_next":
            raise ValueError("shared overlap probe requires GLM-5.3-Flash")
        self.target = target
        self.stats = {}
        self.originals = {}
        for name in ("glm5_next_mlp", "glm5_next_mlp_layer_stationary_tiles"):
            original = getattr(glm5_next, name)
            self.originals[name] = original
            setattr(glm5_next, name, self._wrap(original))

    def _wrap(self, original):
        def wrapped(*args, **kwargs):
            decode = self.target._expert_batch_prefetch_phase == "decode"
            kwargs["shared_expert_overlap"] = decode
            kwargs["shared_overlap_stats"] = self.stats
            return original(*args, **kwargs)
        return wrapped

    def close(self):
        for name, original in self.originals.items():
            setattr(glm5_next, name, original)

from patch_utils import package_root, replace_once


CUSTOM_OPS = package_root("vllm") / "_custom_ops.py"

CLEAN_BLOCK = """\
def copy_blocks(key_caches: List[torch.Tensor],
                value_caches: List[torch.Tensor],
                block_mapping: torch.Tensor) -> None:
    ixf_F.copy_blocks(key_caches, value_caches, block_mapping)
"""

COMPATIBLE_BLOCK = """\
def copy_blocks(key_caches: List[torch.Tensor],
                value_caches: List[torch.Tensor],
                block_mapping: torch.Tensor) -> None:
    # BI100 CoreX 3.2.3 exposes vllm_copy_blocks, not copy_blocks.
    _fn = getattr(ixf_F, "copy_blocks", None) or getattr(
        ixf_F, "vllm_copy_blocks", None)
    if _fn is None:
        raise RuntimeError(
            "ixformer exposes neither copy_blocks nor vllm_copy_blocks")
    _fn(key_caches, value_caches, block_mapping)
"""


replace_once(
    CUSTOM_OPS,
    CLEAN_BLOCK,
    COMPATIBLE_BLOCK,
    required=True,
    already_contains="BI100 CoreX 3.2.3 exposes vllm_copy_blocks",
)
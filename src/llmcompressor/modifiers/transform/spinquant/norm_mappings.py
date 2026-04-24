from collections.abc import Callable
from typing import Dict, List

from loguru import logger
from pydantic import BaseModel, field_validator
from transformers import PreTrainedModel

__all__ = ["infer_norm_mapping_from_model"]


class NormMapping(BaseModel):
    """
    SpinQuant needs to know where every norm layer exists in the model,
    as well as all the subsequent Linear layers the norm passes into.
    This is because the norm layer weights need to normalized before
    transforms can be fused into Linear layers.

    :param norm: name or regex that matches norm layer in model
    :param linears: list of names or regexes of Linear layers that
    receive input from norm.
    """

    norm: str
    linears: List[str]

    @field_validator("linears", mode="before")
    def cast_to_list(cls, value):
        if isinstance(value, str):
            return [value]

        return value


_default_mappings = [
    NormMapping(
        norm="re:.*input_layernorm$",
        linears=["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
    ),
    NormMapping(
        norm="re:.*post_attention_layernorm$",
        linears=["re:.*up_proj$", "re:.*gate_proj$"],
    ),
    NormMapping(
        norm="model.norm",
        linears=["lm_head"],
    ),
]

NormMappingFactory = Callable[[PreTrainedModel], List[NormMapping]]


def _get_nested_attr(obj: object, path: str):
    current = obj
    for part in path.split("."):
        current = getattr(current, part)
    return current


def _get_qwen3_5_text_paths(model: PreTrainedModel) -> tuple[str, str, str, object]:
    for layers_path, embedding_path, norm_path in (
        ("model.layers", "model.embed_tokens", "model.norm"),
        (
            "model.language_model.layers",
            "model.language_model.embed_tokens",
            "model.language_model.norm",
        ),
        ("language_model.layers", "language_model.embed_tokens", "language_model.norm"),
        ("layers", "embed_tokens", "norm"),
    ):
        try:
            layers = _get_nested_attr(model, layers_path)
            _get_nested_attr(model, embedding_path)
            _get_nested_attr(model, norm_path)
            return layers_path, embedding_path, norm_path, layers
        except AttributeError:
            continue

    raise ValueError("Could not locate Qwen3.5 text layers in model")


def _get_qwen3_5_input_linears(layer_prefix: str, layer: object) -> list[str]:
    if hasattr(layer, "self_attn"):
        return [
            f"{layer_prefix}.self_attn.q_proj",
            f"{layer_prefix}.self_attn.k_proj",
            f"{layer_prefix}.self_attn.v_proj",
        ]

    if hasattr(layer, "linear_attn"):
        return [
            f"{layer_prefix}.linear_attn.in_proj_qkv",
            f"{layer_prefix}.linear_attn.in_proj_z",
            f"{layer_prefix}.linear_attn.in_proj_b",
            f"{layer_prefix}.linear_attn.in_proj_a",
        ]

    return []


def _build_qwen3_5_dense_norm_mappings(model: PreTrainedModel) -> List[NormMapping]:
    layers_path, _, norm_path, layers = _get_qwen3_5_text_paths(model)
    mappings: List[NormMapping] = []

    for layer_idx, layer in enumerate(layers):
        layer_prefix = f"{layers_path}.{layer_idx}"
        input_linears = _get_qwen3_5_input_linears(layer_prefix, layer)
        if input_linears:
            mappings.append(
                NormMapping(
                    norm=f"{layer_prefix}.input_layernorm",
                    linears=input_linears,
                )
            )

        mlp = getattr(layer, "mlp", None)
        mlp_linears: list[str] = []
        if mlp is not None:
            if hasattr(mlp, "up_proj"):
                mlp_linears.append(f"{layer_prefix}.mlp.up_proj")
            if hasattr(mlp, "gate_proj"):
                mlp_linears.append(f"{layer_prefix}.mlp.gate_proj")

        if mlp_linears:
            mappings.append(
                NormMapping(
                    norm=f"{layer_prefix}.post_attention_layernorm",
                    linears=mlp_linears,
                )
            )

    mappings.append(NormMapping(norm=norm_path, linears=["lm_head"]))
    return mappings


def _build_qwen3_5_moe_norm_mappings(model: PreTrainedModel) -> List[NormMapping]:
    layers_path, _, norm_path, layers = _get_qwen3_5_text_paths(model)
    mappings: List[NormMapping] = []

    for layer_idx, layer in enumerate(layers):
        layer_prefix = f"{layers_path}.{layer_idx}"
        input_linears = _get_qwen3_5_input_linears(layer_prefix, layer)
        if input_linears:
            mappings.append(
                NormMapping(
                    norm=f"{layer_prefix}.input_layernorm",
                    linears=input_linears,
                )
            )

        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue

        mlp_linears: list[str] = []
        if hasattr(mlp, "gate"):
            mlp_linears.append(f"{layer_prefix}.mlp.gate")

        experts = getattr(mlp, "experts", None)
        if experts is not None:
            try:
                expert_items = list(enumerate(experts))
            except TypeError:
                expert_items = []

            for expert_idx, expert in expert_items:
                if hasattr(expert, "up_proj"):
                    mlp_linears.append(f"{layer_prefix}.mlp.experts.{expert_idx}.up_proj")
                if hasattr(expert, "gate_proj"):
                    mlp_linears.append(f"{layer_prefix}.mlp.experts.{expert_idx}.gate_proj")

        shared_expert = getattr(mlp, "shared_expert", None)
        if shared_expert is not None:
            if hasattr(shared_expert, "up_proj"):
                mlp_linears.append(f"{layer_prefix}.mlp.shared_expert.up_proj")
            if hasattr(shared_expert, "gate_proj"):
                mlp_linears.append(f"{layer_prefix}.mlp.shared_expert.gate_proj")

        if hasattr(mlp, "shared_expert_gate"):
            mlp_linears.append(f"{layer_prefix}.mlp.shared_expert_gate")

        if mlp_linears:
            mappings.append(
                NormMapping(
                    norm=f"{layer_prefix}.post_attention_layernorm",
                    linears=mlp_linears,
                )
            )

    mappings.append(NormMapping(norm=norm_path, linears=["lm_head"]))
    return mappings


NORM_MAPPING_REGISTRY: Dict[str, List[NormMapping]] = {
    "LlamaForCausalLM": _default_mappings,
}

MODEL_AWARE_NORM_MAPPING_REGISTRY: Dict[str, NormMappingFactory] = {
    "Qwen3_5ForCausalLM": _build_qwen3_5_dense_norm_mappings,
    "Qwen3_5ForConditionalGeneration": _build_qwen3_5_dense_norm_mappings,
    "Qwen3_5MoeForCausalLM": _build_qwen3_5_moe_norm_mappings,
    "Qwen3_5MoeForConditionalGeneration": _build_qwen3_5_moe_norm_mappings,
}


def infer_norm_mapping_from_model(model: PreTrainedModel) -> List[NormMapping]:
    architecture = model.__class__.__name__
    if architecture in MODEL_AWARE_NORM_MAPPING_REGISTRY:
        return MODEL_AWARE_NORM_MAPPING_REGISTRY[architecture](model)

    if architecture not in NORM_MAPPING_REGISTRY:
        logger.info(
            f"Unrecognized model architecture {architecture}. "
            "Falling back to default mappings"
        )

    return NORM_MAPPING_REGISTRY.get(architecture, _default_mappings)

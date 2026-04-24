from collections.abc import Callable
from typing import Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field, field_validator
from transformers import PreTrainedModel

__all__ = ["SpinQuantMapping", "infer_mapping_from_model"]


class SpinQuantMapping(BaseModel):
    """
    SpinQuant needs to know the entire architecture of the model,
    as R1, R2, R3, and R4 rotations need to be applied to specific
    layers (https://arxiv.org/pdf/2405.16406 Fig. 1).

    :param embedding: name or regex of embedding layer
    :param attn: names or regexes of attention blocks in decoder layers that support
        SpinQuant's online attention rotations (R3)
    :param attn_q: names or regexes of q_proj layers in attention blocks
    :param attn_k: names or regexes of k_proj layers in attention blocks
    :param attn_v: names or regexes of v_proj layers in attention blocks
    :param attn_o: names or regexes of o_proj layers in attention blocks
    :param attn_head_dim: head_dim of the attention module, needed
        because R2 needs to be applied "head-wisely" to v_proj and
        o_proj
    :param token_mixer_in: names or regexes for every token-mixer projection that
        directly consumes the hidden-state stream. If not provided, defaults to
        attn_q/attn_k/attn_v.
    :param token_mixer_out: names or regexes for every token-mixer projection that
        writes back to the hidden-state stream. If not provided, defaults to attn_o.
    :param mlp_in: list of names or regexes for the mlp blocks that
        receive the input to the MLP block, usually up_proj and gate_proj
    :param mlp_out: list of names or regexes for the mlp blocks that
        constitute the output of the MLP block, usually down_proj
    """

    embedding: str

    attn: List[str]
    attn_q: List[str]
    attn_k: List[str]
    attn_v: List[str]
    attn_o: List[str]
    attn_head_dim: Optional[int] = Field(default=None)
    token_mixer_in: Optional[List[str]] = Field(default=None)
    token_mixer_out: Optional[List[str]] = Field(default=None)

    mlp_in: List[str]  # up_proj, gate_proj
    mlp_out: List[str]  # down_proj

    lm_head: str

    @field_validator(
        "attn",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_o",
        "token_mixer_in",
        "token_mixer_out",
        "mlp_in",
        "mlp_out",
        mode="before",
    )
    def cast_to_list(cls, value):
        if value is None:
            return None

        if isinstance(value, str):
            return [value]

        return value


_default_mappings = SpinQuantMapping(
    embedding="re:.*embed_tokens$",
    attn="re:.*self_attn$",
    attn_q="re:.*q_proj$",
    attn_k="re:.*k_proj$",
    attn_v="re:.*v_proj$",
    attn_o="re:.*o_proj$",
    mlp_in=["re:.*up_proj$", "re:.*gate_proj$"],
    mlp_out="re:.*down_proj$",
    lm_head="lm_head",
)


SpinQuantMappingFactory = Callable[[PreTrainedModel], SpinQuantMapping]


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


def _collect_qwen3_5_token_mixer(
    layer_prefix: str, layer: object
) -> tuple[
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
    list[str],
]:
    attn_modules: list[str] = []
    attn_q: list[str] = []
    attn_k: list[str] = []
    attn_v: list[str] = []
    attn_o: list[str] = []
    token_mixer_in: list[str] = []
    token_mixer_out: list[str] = []

    if hasattr(layer, "self_attn"):
        attn_prefix = f"{layer_prefix}.self_attn"
        attn_modules.append(attn_prefix)
        attn_q.append(f"{attn_prefix}.q_proj")
        attn_k.append(f"{attn_prefix}.k_proj")
        attn_v.append(f"{attn_prefix}.v_proj")
        attn_o.append(f"{attn_prefix}.o_proj")
        token_mixer_in.extend([*attn_q[-1:], *attn_k[-1:], *attn_v[-1:]])
        token_mixer_out.extend(attn_o[-1:])
    elif hasattr(layer, "linear_attn"):
        linear_attn_prefix = f"{layer_prefix}.linear_attn"
        token_mixer_in.extend(
            [
                f"{linear_attn_prefix}.in_proj_qkv",
                f"{linear_attn_prefix}.in_proj_z",
                f"{linear_attn_prefix}.in_proj_b",
                f"{linear_attn_prefix}.in_proj_a",
            ]
        )
        token_mixer_out.append(f"{linear_attn_prefix}.out_proj")

    return (
        attn_modules,
        attn_q,
        attn_k,
        attn_v,
        attn_o,
        token_mixer_in,
        token_mixer_out,
    )


def _build_qwen3_5_dense_mapping(model: PreTrainedModel) -> SpinQuantMapping:
    layers_path, embedding_path, _, layers = _get_qwen3_5_text_paths(model)

    attn_modules: list[str] = []
    attn_q: list[str] = []
    attn_k: list[str] = []
    attn_v: list[str] = []
    attn_o: list[str] = []
    token_mixer_in: list[str] = []
    token_mixer_out: list[str] = []
    mlp_in: list[str] = []
    mlp_out: list[str] = []

    for layer_idx, layer in enumerate(layers):
        layer_prefix = f"{layers_path}.{layer_idx}"
        (
            layer_attn_modules,
            layer_attn_q,
            layer_attn_k,
            layer_attn_v,
            layer_attn_o,
            layer_token_mixer_in,
            layer_token_mixer_out,
        ) = _collect_qwen3_5_token_mixer(layer_prefix, layer)
        attn_modules.extend(layer_attn_modules)
        attn_q.extend(layer_attn_q)
        attn_k.extend(layer_attn_k)
        attn_v.extend(layer_attn_v)
        attn_o.extend(layer_attn_o)
        token_mixer_in.extend(layer_token_mixer_in)
        token_mixer_out.extend(layer_token_mixer_out)

        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            if hasattr(mlp, "up_proj"):
                mlp_in.append(f"{layer_prefix}.mlp.up_proj")
            if hasattr(mlp, "gate_proj"):
                mlp_in.append(f"{layer_prefix}.mlp.gate_proj")
            if hasattr(mlp, "down_proj"):
                mlp_out.append(f"{layer_prefix}.mlp.down_proj")

    return SpinQuantMapping(
        embedding=embedding_path,
        attn=attn_modules,
        attn_q=attn_q,
        attn_k=attn_k,
        attn_v=attn_v,
        attn_o=attn_o,
        token_mixer_in=token_mixer_in,
        token_mixer_out=token_mixer_out,
        mlp_in=mlp_in,
        mlp_out=mlp_out,
        lm_head="lm_head",
    )


def _build_qwen3_5_moe_mapping(model: PreTrainedModel) -> SpinQuantMapping:
    layers_path, embedding_path, _, layers = _get_qwen3_5_text_paths(model)

    attn_modules: list[str] = []
    attn_q: list[str] = []
    attn_k: list[str] = []
    attn_v: list[str] = []
    attn_o: list[str] = []
    token_mixer_in: list[str] = []
    token_mixer_out: list[str] = []
    mlp_in: list[str] = []
    mlp_out: list[str] = []

    for layer_idx, layer in enumerate(layers):
        layer_prefix = f"{layers_path}.{layer_idx}"
        (
            layer_attn_modules,
            layer_attn_q,
            layer_attn_k,
            layer_attn_v,
            layer_attn_o,
            layer_token_mixer_in,
            layer_token_mixer_out,
        ) = _collect_qwen3_5_token_mixer(layer_prefix, layer)
        attn_modules.extend(layer_attn_modules)
        attn_q.extend(layer_attn_q)
        attn_k.extend(layer_attn_k)
        attn_v.extend(layer_attn_v)
        attn_o.extend(layer_attn_o)
        token_mixer_in.extend(layer_token_mixer_in)
        token_mixer_out.extend(layer_token_mixer_out)

        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue

        if hasattr(mlp, "gate"):
            mlp_in.append(f"{layer_prefix}.mlp.gate")

        experts = getattr(mlp, "experts", None)
        if experts is not None:
            try:
                expert_items = list(enumerate(experts))
            except TypeError:
                expert_items = []

            for expert_idx, expert in expert_items:
                if hasattr(expert, "gate_proj"):
                    mlp_in.append(f"{layer_prefix}.mlp.experts.{expert_idx}.gate_proj")
                if hasattr(expert, "up_proj"):
                    mlp_in.append(f"{layer_prefix}.mlp.experts.{expert_idx}.up_proj")
                if hasattr(expert, "down_proj"):
                    mlp_out.append(f"{layer_prefix}.mlp.experts.{expert_idx}.down_proj")

        shared_expert = getattr(mlp, "shared_expert", None)
        if shared_expert is not None:
            if hasattr(shared_expert, "gate_proj"):
                mlp_in.append(f"{layer_prefix}.mlp.shared_expert.gate_proj")
            if hasattr(shared_expert, "up_proj"):
                mlp_in.append(f"{layer_prefix}.mlp.shared_expert.up_proj")
            if hasattr(shared_expert, "down_proj"):
                mlp_out.append(f"{layer_prefix}.mlp.shared_expert.down_proj")

        if hasattr(mlp, "shared_expert_gate"):
            mlp_in.append(f"{layer_prefix}.mlp.shared_expert_gate")

    return SpinQuantMapping(
        embedding=embedding_path,
        attn=attn_modules,
        attn_q=attn_q,
        attn_k=attn_k,
        attn_v=attn_v,
        attn_o=attn_o,
        token_mixer_in=token_mixer_in,
        token_mixer_out=token_mixer_out,
        mlp_in=mlp_in,
        mlp_out=mlp_out,
        lm_head="lm_head",
    )


SPINQUANT_MAPPING_REGISTRY: Dict[str, SpinQuantMapping] = {
    "LlamaForCausalLM": _default_mappings,
}

MODEL_AWARE_SPINQUANT_MAPPING_REGISTRY: Dict[str, SpinQuantMappingFactory] = {
    "Qwen3_5ForCausalLM": _build_qwen3_5_dense_mapping,
    "Qwen3_5ForConditionalGeneration": _build_qwen3_5_dense_mapping,
    "Qwen3_5MoeForCausalLM": _build_qwen3_5_moe_mapping,
    "Qwen3_5MoeForConditionalGeneration": _build_qwen3_5_moe_mapping,
}


def infer_mapping_from_model(model: PreTrainedModel) -> SpinQuantMapping:
    architecture = model.__class__.__name__
    if architecture in MODEL_AWARE_SPINQUANT_MAPPING_REGISTRY:
        return MODEL_AWARE_SPINQUANT_MAPPING_REGISTRY[architecture](model)

    if architecture not in SPINQUANT_MAPPING_REGISTRY:
        logger.info(
            f"Unrecognized model architecture {architecture}. "
            "Falling back to default mappings"
        )

    return SPINQUANT_MAPPING_REGISTRY.get(architecture, _default_mappings)

import functools
from collections import namedtuple
from collections.abc import Callable

from loguru import logger

__all__ = [
    "get_layer_mappings_from_architecture",
    "MAPPINGS_REGISTRY",
    "DEFAULT_SMOOTHQUANT_MAPPINGS",
]

LayerMapType = tuple[list[str], str]
LayerMap: LayerMapType = namedtuple("LayerMap", ["balance_layers", "smooth_layers"])

DEFAULT_SMOOTHQUANT_MAPPINGS: list[LayerMap] = [
    LayerMap(
        balance_layers=["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
        smooth_layers="re:.*input_layernorm",
    ),
    LayerMap(
        balance_layers=["re:.*gate_proj", "re:.*up_proj"],
        smooth_layers="re:.*post_attention_layernorm",
    ),
]
MIXTRAL_SMOOTHQUANT_MAPPINGS: list[LayerMap] = [
    LayerMap(
        balance_layers=["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"],
        smooth_layers="re:.*input_layernorm",
    ),
]
BLOOM_SMOOTHQUANT_MAPPINGS: list[LayerMap] = [
    LayerMap(
        balance_layers=["re:.*query_key_value"],
        smooth_layers="re:.*input_layernorm",
    ),
    LayerMap(
        balance_layers=["re:.*dense_h_to_4h"],
        smooth_layers="re:.*post_attention_layernorm",
    ),
]
PHI3_VISION_SMOOTHQUANT_MAPPINGS: list[LayerMap] = [
    LayerMap(
        balance_layers=["re:.*qkv_proj"],
        smooth_layers="re:.*input_layernorm",
    ),
    LayerMap(
        balance_layers=["re:.*gate_up_proj"],
        smooth_layers="re:.*post_attention_layernorm",
    ),
]
WHISPER_V2_SMOOTHQUANT_MAPPINGS: list[LayerMap] = [
    LayerMap(
        balance_layers=["re:.*k_proj", "re:.*v_proj", "re:.*q_proj"],
        smooth_layers="re:.*self_attn_layer_norm",
    ),
    LayerMap(
        balance_layers=["re:.*fc1"],
        smooth_layers="re:.*final_layer_norm",
    ),
]

DEEPSEEK_V2_SMOOTHQUANT_MAPPINGS: list[LayerMap] = [
    LayerMap(
        balance_layers=["re:.*q(_a)?_proj$", "re:.*kv_a_proj_with_mqa"],
        smooth_layers="re:.*input_layernorm",
    ),
]

AFMOE_SMOOTHQUANT_MAPPINGS: list[LayerMap] = [
    LayerMap(
        balance_layers=[
            "re:.*self_attn\\.q_proj",
            "re:.*self_attn\\.k_proj",
            "re:.*self_attn\\.v_proj",
            "re:.*self_attn\\.gate_proj",
        ],
        smooth_layers="re:.*input_layernorm",
    ),
    LayerMap(
        balance_layers=["re:.*mlp.*gate_proj", "re:.*mlp.*up_proj"],
        smooth_layers="re:.*pre_mlp_layernorm",
    ),
]

ModelAwareLayerMapFactory = Callable[[object], list[LayerMap]]


def _get_nested_attr(obj: object, path: str):
    current = obj
    for part in path.split("."):
        current = getattr(current, part)
    return current


def _get_qwen3_5_layer_container(model: object) -> tuple[str, object]:
    for path in (
        "model.layers",
        "model.language_model.layers",
        "language_model.layers",
        "layers",
    ):
        try:
            return path, _get_nested_attr(model, path)
        except AttributeError:
            continue

    raise ValueError("Could not locate decoder layers for Qwen3.5 model")


def _get_qwen3_5_input_balance_layers(
    layer_prefix: str, layer: object
) -> list[str] | None:
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

    return None


def _get_qwen3_5_dense_layer_mappings(model: object) -> list[LayerMap]:
    layers_path, layers = _get_qwen3_5_layer_container(model)
    mappings: list[LayerMap] = []

    for layer_idx, layer in enumerate(layers):
        layer_prefix = f"{layers_path}.{layer_idx}"

        input_balance_layers = _get_qwen3_5_input_balance_layers(layer_prefix, layer)
        if input_balance_layers:
            mappings.append(
                LayerMap(
                    balance_layers=input_balance_layers,
                    smooth_layers=f"{layer_prefix}.input_layernorm",
                )
            )

        mlp = getattr(layer, "mlp", None)
        mlp_balance_layers: list[str] = []
        if mlp is not None:
            if hasattr(mlp, "gate_proj"):
                mlp_balance_layers.append(f"{layer_prefix}.mlp.gate_proj")
            if hasattr(mlp, "up_proj"):
                mlp_balance_layers.append(f"{layer_prefix}.mlp.up_proj")

        if mlp_balance_layers:
            mappings.append(
                LayerMap(
                    balance_layers=mlp_balance_layers,
                    smooth_layers=f"{layer_prefix}.post_attention_layernorm",
                )
            )

    return mappings


def _get_qwen3_5_moe_layer_mappings(model: object) -> list[LayerMap]:
    layers_path, layers = _get_qwen3_5_layer_container(model)
    mappings: list[LayerMap] = []

    for layer_idx, layer in enumerate(layers):
        layer_prefix = f"{layers_path}.{layer_idx}"

        input_balance_layers = _get_qwen3_5_input_balance_layers(layer_prefix, layer)
        if input_balance_layers:
            mappings.append(
                LayerMap(
                    balance_layers=input_balance_layers,
                    smooth_layers=f"{layer_prefix}.input_layernorm",
                )
            )

        mlp_balance_layers: list[str] = []
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            if hasattr(mlp, "gate"):
                mlp_balance_layers.append(f"{layer_prefix}.mlp.gate")

            experts = getattr(mlp, "experts", None)
            if experts is not None:
                for expert_idx, expert in enumerate(experts):
                    if hasattr(expert, "gate_proj"):
                        mlp_balance_layers.append(
                            f"{layer_prefix}.mlp.experts.{expert_idx}.gate_proj"
                        )
                    if hasattr(expert, "up_proj"):
                        mlp_balance_layers.append(
                            f"{layer_prefix}.mlp.experts.{expert_idx}.up_proj"
                        )

            shared_expert = getattr(mlp, "shared_expert", None)
            if shared_expert is not None:
                if hasattr(shared_expert, "gate_proj"):
                    mlp_balance_layers.append(
                        f"{layer_prefix}.mlp.shared_expert.gate_proj"
                    )
                if hasattr(shared_expert, "up_proj"):
                    mlp_balance_layers.append(
                        f"{layer_prefix}.mlp.shared_expert.up_proj"
                    )

            if hasattr(mlp, "shared_expert_gate"):
                mlp_balance_layers.append(f"{layer_prefix}.mlp.shared_expert_gate")

        if mlp_balance_layers:
            mappings.append(
                LayerMap(
                    balance_layers=mlp_balance_layers,
                    smooth_layers=f"{layer_prefix}.post_attention_layernorm",
                )
            )

    return mappings


MODEL_AWARE_MAPPINGS_REGISTRY: dict[str, ModelAwareLayerMapFactory] = {
    "Qwen3_5ForCausalLM": _get_qwen3_5_dense_layer_mappings,
    "Qwen3_5ForConditionalGeneration": _get_qwen3_5_dense_layer_mappings,
    "Qwen3_5MoeForCausalLM": _get_qwen3_5_moe_layer_mappings,
    "Qwen3_5MoeForConditionalGeneration": _get_qwen3_5_moe_layer_mappings,
}


# Registry of layer mappings for different architectures
#   Add more mappings here
MAPPINGS_REGISTRY: dict[str, list[LayerMap]] = {
    "BloomForCausalLM": BLOOM_SMOOTHQUANT_MAPPINGS,
    "ChatGLMForConditionalGeneration": BLOOM_SMOOTHQUANT_MAPPINGS,
    "DeepseekV2ForCausalLM": DEEPSEEK_V2_SMOOTHQUANT_MAPPINGS,
    "Gemma2ForCausalLM": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "Gemma3ForCausalLM": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "Gemma3ForConditionalGeneration": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "Glm4MoeForCausalLM": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "GlmMoeDsaForCausalLM": DEEPSEEK_V2_SMOOTHQUANT_MAPPINGS,
    "Llama4ForConditionalGeneration": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "LlamaForCausalLM": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "Mistral3ForConditionalGeneration": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "MistralForCausalLM": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "MixtralForCausalLM": MIXTRAL_SMOOTHQUANT_MAPPINGS,
    "Phi3VForCausalLM": PHI3_VISION_SMOOTHQUANT_MAPPINGS,
    "Qwen2ForCausalLM": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "Qwen3ForCausalLM": DEFAULT_SMOOTHQUANT_MAPPINGS,
    "WhisperForConditionalGeneration": WHISPER_V2_SMOOTHQUANT_MAPPINGS,
    "AfmoeForCausalLM": AFMOE_SMOOTHQUANT_MAPPINGS,
}


def get_layer_mappings_from_architecture(
    architecture: str, model: object | None = None
) -> list[LayerMap]:
    """
    :param architecture: str: The architecture of the model
    :return: list: The layer mappings for the given architecture
    """

    if model is not None and architecture in MODEL_AWARE_MAPPINGS_REGISTRY:
        return MODEL_AWARE_MAPPINGS_REGISTRY[architecture](model)

    if architecture not in MAPPINGS_REGISTRY:
        logger.info(
            f"Architecture {architecture} not found in mappings. "
            f"Using default mappings: {DEFAULT_SMOOTHQUANT_MAPPINGS}"
        )

    return MAPPINGS_REGISTRY.get(architecture, DEFAULT_SMOOTHQUANT_MAPPINGS)


def handle_mapping_resolution_errors(func):
    """
    Decorator to catch any errors that occur when resolving mappings and provide a
    helpful error message to the user pointing them to the README
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as original_exception:
            readme_location = (
                "https://github.com/vllm-project/llm-compressor/tree/main/"
                "src/llmcompressor/modifiers/transform/smoothquant"
            )
            raise RuntimeError(
                f"Error resolving mappings for given architecture."
                f"Please refer to the README at {readme_location} for more information."
            ) from original_exception

    return wrapper

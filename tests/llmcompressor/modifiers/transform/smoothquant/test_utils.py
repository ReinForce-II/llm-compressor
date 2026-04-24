from unittest.mock import patch

import pytest
import torch
from compressed_tensors.utils import match_modules_set
from torch.utils._pytree import tree_leaves

from llmcompressor.modifiers.transform.smoothquant.utils import (
    get_layer_mappings_from_architecture,
    handle_mapping_resolution_errors,
)

smoothquant_utils = "llmcompressor.modifiers.transform.smoothquant.utils"


@pytest.mark.unit
def test_handle_mapping_resolution_errors():
    README_LOCATION = (
        "https://github.com/vllm-project/llm-compressor/tree/main/"
        "src/llmcompressor/modifiers/transform/smoothquant"
    )

    @handle_mapping_resolution_errors
    def func_that_raises_exception():
        raise ValueError("An error occurred")

    with pytest.raises(RuntimeError) as excinfo:
        func_that_raises_exception()

    assert "Error resolving mappings for given architecture." in str(excinfo.value)
    assert "Please refer to the README at" in str(excinfo.value)
    assert README_LOCATION in str(excinfo.value)


@pytest.mark.unit
@patch(
    f"{smoothquant_utils}.MAPPINGS_REGISTRY", {"arch1": "mapping1", "arch2": "mapping2"}
)
@patch(f"{smoothquant_utils}.DEFAULT_SMOOTHQUANT_MAPPINGS", "default_mapping")
def test_get_layer_mappings_from_architecture():
    # Test when architecture is in MAPPINGS_REGISTRY
    assert get_layer_mappings_from_architecture("arch1") == "mapping1"

    # Test when architecture is not in MAPPINGS_REGISTRY
    assert get_layer_mappings_from_architecture("arch3") == "default_mapping"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("architecture", "prefix", "nested_under_language_model"),
    [
        ("Qwen3_5ForCausalLM", "model.layers", False),
        ("Qwen3_5ForConditionalGeneration", "model.language_model.layers", True),
    ],
)
def test_get_layer_mappings_from_architecture_qwen3_5_dense(
    architecture, prefix, nested_under_language_model
):
    hidden_size = 16
    intermediate_size = 32

    def make_layers():
        return torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "input_layernorm": torch.nn.LayerNorm(hidden_size),
                        "self_attn": torch.nn.ModuleDict(
                            {
                                "q_proj": torch.nn.Linear(
                                    hidden_size, hidden_size, bias=False
                                ),
                                "k_proj": torch.nn.Linear(
                                    hidden_size, hidden_size, bias=False
                                ),
                                "v_proj": torch.nn.Linear(
                                    hidden_size, hidden_size, bias=False
                                ),
                            }
                        ),
                        "post_attention_layernorm": torch.nn.LayerNorm(hidden_size),
                        "mlp": torch.nn.ModuleDict(
                            {
                                "gate_proj": torch.nn.Linear(
                                    hidden_size, intermediate_size, bias=False
                                ),
                                "up_proj": torch.nn.Linear(
                                    hidden_size, intermediate_size, bias=False
                                ),
                            }
                        ),
                    }
                ),
                torch.nn.ModuleDict(
                    {
                        "input_layernorm": torch.nn.LayerNorm(hidden_size),
                        "linear_attn": torch.nn.ModuleDict(
                            {
                                "in_proj_qkv": torch.nn.Linear(
                                    hidden_size, hidden_size, bias=False
                                ),
                                "in_proj_z": torch.nn.Linear(
                                    hidden_size, hidden_size, bias=False
                                ),
                                "in_proj_b": torch.nn.Linear(hidden_size, 4, bias=False),
                                "in_proj_a": torch.nn.Linear(hidden_size, 4, bias=False),
                            }
                        ),
                        "post_attention_layernorm": torch.nn.LayerNorm(hidden_size),
                        "mlp": torch.nn.ModuleDict(
                            {
                                "gate_proj": torch.nn.Linear(
                                    hidden_size, intermediate_size, bias=False
                                ),
                                "up_proj": torch.nn.Linear(
                                    hidden_size, intermediate_size, bias=False
                                ),
                            }
                        ),
                    }
                ),
            ]
        )

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    if nested_under_language_model:
        model.model.language_model = torch.nn.Module()
        model.model.language_model.layers = make_layers()
    else:
        model.model.layers = make_layers()

    mappings = get_layer_mappings_from_architecture(architecture, model=model)

    assert [mapping.smooth_layers for mapping in mappings] == [
        f"{prefix}.0.input_layernorm",
        f"{prefix}.0.post_attention_layernorm",
        f"{prefix}.1.input_layernorm",
        f"{prefix}.1.post_attention_layernorm",
    ]
    assert mappings[1].balance_layers == [
        f"{prefix}.0.mlp.gate_proj",
        f"{prefix}.0.mlp.up_proj",
    ]
    assert mappings[3].balance_layers == [
        f"{prefix}.1.mlp.gate_proj",
        f"{prefix}.1.mlp.up_proj",
    ]

    module_to_name = {module: name for name, module in model.named_modules()}
    summary = {}
    for mapping in mappings:
        for *nested_balance_layers, smooth_layers in match_modules_set(
            model, tree_leaves(mapping)
        ):
            summary[module_to_name[smooth_layers[0]]] = len(
                tree_leaves(nested_balance_layers)
            )

    assert summary == {
        f"{prefix}.0.input_layernorm": 3,
        f"{prefix}.0.post_attention_layernorm": 2,
        f"{prefix}.1.input_layernorm": 4,
        f"{prefix}.1.post_attention_layernorm": 2,
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("architecture", "prefix", "nested_under_language_model"),
    [
        ("Qwen3_5MoeForCausalLM", "model.layers", False),
        (
            "Qwen3_5MoeForConditionalGeneration",
            "model.language_model.layers",
            True,
        ),
    ],
)
def test_get_layer_mappings_from_architecture_qwen3_5_moe(
    architecture, prefix, nested_under_language_model
):
    hidden_size = 16
    num_experts = 2

    def make_mlp():
        return torch.nn.ModuleDict(
            {
                "gate": torch.nn.Linear(hidden_size, num_experts, bias=False),
                "experts": torch.nn.ModuleList(
                    [
                        torch.nn.ModuleDict(
                            {
                                "gate_proj": torch.nn.Linear(
                                    hidden_size, hidden_size, bias=False
                                ),
                                "up_proj": torch.nn.Linear(
                                    hidden_size, hidden_size, bias=False
                                ),
                            }
                        )
                        for _ in range(num_experts)
                    ]
                ),
                "shared_expert": torch.nn.ModuleDict(
                    {
                        "gate_proj": torch.nn.Linear(
                            hidden_size, hidden_size, bias=False
                        ),
                        "up_proj": torch.nn.Linear(
                            hidden_size, hidden_size, bias=False
                        ),
                    }
                ),
                "shared_expert_gate": torch.nn.Linear(hidden_size, 1, bias=False),
            }
        )

    def make_layers():
        return torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "input_layernorm": torch.nn.LayerNorm(hidden_size),
                        "self_attn": torch.nn.ModuleDict(
                            {
                                "q_proj": torch.nn.Linear(
                                    hidden_size, hidden_size, bias=False
                                ),
                                "k_proj": torch.nn.Linear(
                                    hidden_size, hidden_size, bias=False
                                ),
                                "v_proj": torch.nn.Linear(
                                    hidden_size, hidden_size, bias=False
                                ),
                            }
                        ),
                        "post_attention_layernorm": torch.nn.LayerNorm(hidden_size),
                        "mlp": make_mlp(),
                    }
                ),
                torch.nn.ModuleDict(
                    {
                        "input_layernorm": torch.nn.LayerNorm(hidden_size),
                        "linear_attn": torch.nn.ModuleDict(
                            {
                                "in_proj_qkv": torch.nn.Linear(
                                    hidden_size, hidden_size, bias=False
                                ),
                                "in_proj_z": torch.nn.Linear(
                                    hidden_size, hidden_size, bias=False
                                ),
                                "in_proj_b": torch.nn.Linear(hidden_size, 4, bias=False),
                                "in_proj_a": torch.nn.Linear(hidden_size, 4, bias=False),
                            }
                        ),
                        "post_attention_layernorm": torch.nn.LayerNorm(hidden_size),
                        "mlp": make_mlp(),
                    }
                ),
            ]
        )

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    if nested_under_language_model:
        model.model.language_model = torch.nn.Module()
        model.model.language_model.layers = make_layers()
    else:
        model.model.layers = make_layers()

    mappings = get_layer_mappings_from_architecture(architecture, model=model)

    assert [mapping.smooth_layers for mapping in mappings] == [
        f"{prefix}.0.input_layernorm",
        f"{prefix}.0.post_attention_layernorm",
        f"{prefix}.1.input_layernorm",
        f"{prefix}.1.post_attention_layernorm",
    ]
    assert f"{prefix}.0.mlp.gate" in mappings[1].balance_layers
    assert f"{prefix}.0.mlp.shared_expert_gate" in mappings[1].balance_layers

    module_to_name = {module: name for name, module in model.named_modules()}
    summary = {}
    for mapping in mappings:
        for *nested_balance_layers, smooth_layers in match_modules_set(
            model, tree_leaves(mapping)
        ):
            summary[module_to_name[smooth_layers[0]]] = len(
                tree_leaves(nested_balance_layers)
            )

    assert summary == {
        f"{prefix}.0.input_layernorm": 3,
        f"{prefix}.0.post_attention_layernorm": 8,
        f"{prefix}.1.input_layernorm": 4,
        f"{prefix}.1.post_attention_layernorm": 8,
    }

import pytest
import torch
from compressed_tensors import match_modules_set
from torch.utils._pytree import tree_leaves

from llmcompressor.modifiers.transform.spinquant.mappings import (
    infer_mapping_from_model,
)
from llmcompressor.modifiers.transform.spinquant.norm_mappings import (
    infer_norm_mapping_from_model,
)


class Qwen3_5ForCausalLM(torch.nn.Module):
    pass


class Qwen3_5ForConditionalGeneration(torch.nn.Module):
    pass


class Qwen3_5MoeForCausalLM(torch.nn.Module):
    pass


class Qwen3_5MoeForConditionalGeneration(torch.nn.Module):
    pass


def _make_router_gate(hidden_size: int, num_experts: int) -> torch.nn.Module:
    gate = torch.nn.Module()
    gate.register_parameter(
        "weight", torch.nn.Parameter(torch.randn(num_experts, hidden_size))
    )
    return gate


def _make_dense_layers(hidden_size: int, intermediate_size: int) -> torch.nn.ModuleList:
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
                            "o_proj": torch.nn.Linear(
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
                            "down_proj": torch.nn.Linear(
                                intermediate_size, hidden_size, bias=False
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
                            "out_proj": torch.nn.Linear(
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
                            "down_proj": torch.nn.Linear(
                                intermediate_size, hidden_size, bias=False
                            ),
                        }
                    ),
                }
            ),
        ]
    )


def _make_moe_layers(hidden_size: int, num_experts: int) -> torch.nn.ModuleList:
    def make_layer(token_mixer: torch.nn.ModuleDict):
        return torch.nn.ModuleDict(
            {
                "input_layernorm": torch.nn.LayerNorm(hidden_size),
                **token_mixer,
                "post_attention_layernorm": torch.nn.LayerNorm(hidden_size),
                "mlp": torch.nn.ModuleDict(
                    {
                        "gate": _make_router_gate(hidden_size, num_experts),
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
                                        "down_proj": torch.nn.Linear(
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
                                "down_proj": torch.nn.Linear(
                                    hidden_size, hidden_size, bias=False
                                ),
                            }
                        ),
                        "shared_expert_gate": torch.nn.Linear(
                            hidden_size, 1, bias=False
                        ),
                    }
                ),
            }
        )

    return torch.nn.ModuleList(
        [
            make_layer(
                torch.nn.ModuleDict(
                    {
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
                                "o_proj": torch.nn.Linear(
                                    hidden_size, hidden_size, bias=False
                                ),
                            }
                        )
                    }
                )
            ),
            make_layer(
                torch.nn.ModuleDict(
                    {
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
                                "out_proj": torch.nn.Linear(
                                    hidden_size, hidden_size, bias=False
                                ),
                            }
                        )
                    }
                )
            ),
        ]
    )


def _build_model(model_cls: type[torch.nn.Module], nested_under_language_model: bool):
    hidden_size = 16
    intermediate_size = 32
    num_experts = 2

    model = model_cls()
    model.model = torch.nn.Module()

    if model_cls.__name__.startswith("Qwen3_5Moe"):
        layers = _make_moe_layers(hidden_size, num_experts)
    else:
        layers = _make_dense_layers(hidden_size, intermediate_size)

    if nested_under_language_model:
        model.model.language_model = torch.nn.Module()
        model.model.language_model.layers = layers
        model.model.language_model.embed_tokens = torch.nn.Embedding(8, hidden_size)
        model.model.language_model.norm = torch.nn.LayerNorm(hidden_size)
        prefix = "model.language_model.layers"
        embed_path = "model.language_model.embed_tokens"
        norm_path = "model.language_model.norm"
    else:
        model.model.layers = layers
        model.model.embed_tokens = torch.nn.Embedding(8, hidden_size)
        model.model.norm = torch.nn.LayerNorm(hidden_size)
        prefix = "model.layers"
        embed_path = "model.embed_tokens"
        norm_path = "model.norm"

    model.lm_head = torch.nn.Linear(hidden_size, 8, bias=False)

    return model, prefix, embed_path, norm_path


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_cls", "nested_under_language_model", "expected_mlp_in", "expected_mlp_out"),
    [
        (Qwen3_5ForCausalLM, False, 4, 2),
        (Qwen3_5ForConditionalGeneration, True, 4, 2),
        (Qwen3_5MoeForCausalLM, False, 16, 6),
        (Qwen3_5MoeForConditionalGeneration, True, 16, 6),
    ],
)
def test_infer_spinquant_mapping_for_qwen3_5(
    model_cls, nested_under_language_model, expected_mlp_in, expected_mlp_out
):
    model, prefix, embed_path, _ = _build_model(model_cls, nested_under_language_model)

    mapping = infer_mapping_from_model(model)

    assert mapping.embedding == embed_path
    assert mapping.lm_head == "lm_head"
    assert mapping.attn == [f"{prefix}.0.self_attn"]
    assert mapping.attn_q == [f"{prefix}.0.self_attn.q_proj"]
    assert mapping.attn_k == [f"{prefix}.0.self_attn.k_proj"]
    assert mapping.attn_v == [f"{prefix}.0.self_attn.v_proj"]
    assert mapping.attn_o == [f"{prefix}.0.self_attn.o_proj"]
    assert mapping.token_mixer_out == [
        f"{prefix}.0.self_attn.o_proj",
        f"{prefix}.1.linear_attn.out_proj",
    ]
    assert mapping.token_mixer_in == [
        f"{prefix}.0.self_attn.q_proj",
        f"{prefix}.0.self_attn.k_proj",
        f"{prefix}.0.self_attn.v_proj",
        f"{prefix}.1.linear_attn.in_proj_qkv",
        f"{prefix}.1.linear_attn.in_proj_z",
        f"{prefix}.1.linear_attn.in_proj_b",
        f"{prefix}.1.linear_attn.in_proj_a",
    ]
    assert len(mapping.mlp_in) == expected_mlp_in
    assert len(mapping.mlp_out) == expected_mlp_out

    if model_cls.__name__.startswith("Qwen3_5Moe"):
        assert f"{prefix}.0.mlp.gate" in mapping.mlp_in
        assert f"{prefix}.1.mlp.shared_expert_gate" in mapping.mlp_in
        assert f"{prefix}.0.mlp.experts.0.down_proj" in mapping.mlp_out
        assert f"{prefix}.1.mlp.shared_expert.down_proj" in mapping.mlp_out
    else:
        assert mapping.mlp_in == [
            f"{prefix}.0.mlp.up_proj",
            f"{prefix}.0.mlp.gate_proj",
            f"{prefix}.1.mlp.up_proj",
            f"{prefix}.1.mlp.gate_proj",
        ]
        assert mapping.mlp_out == [
            f"{prefix}.0.mlp.down_proj",
            f"{prefix}.1.mlp.down_proj",
        ]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_cls", "nested_under_language_model", "expected_summary"),
    [
        (
            Qwen3_5ForCausalLM,
            False,
            {
                "layer0_input": 3,
                "layer0_post": 2,
                "layer1_input": 4,
                "layer1_post": 2,
                "final_norm": 1,
            },
        ),
        (
            Qwen3_5ForConditionalGeneration,
            True,
            {
                "layer0_input": 3,
                "layer0_post": 2,
                "layer1_input": 4,
                "layer1_post": 2,
                "final_norm": 1,
            },
        ),
        (
            Qwen3_5MoeForCausalLM,
            False,
            {
                "layer0_input": 3,
                "layer0_post": 8,
                "layer1_input": 4,
                "layer1_post": 8,
                "final_norm": 1,
            },
        ),
        (
            Qwen3_5MoeForConditionalGeneration,
            True,
            {
                "layer0_input": 3,
                "layer0_post": 8,
                "layer1_input": 4,
                "layer1_post": 8,
                "final_norm": 1,
            },
        ),
    ],
)
def test_infer_spinquant_norm_mapping_for_qwen3_5(
    model_cls, nested_under_language_model, expected_summary
):
    model, prefix, _, norm_path = _build_model(model_cls, nested_under_language_model)

    norm_mappings = infer_norm_mapping_from_model(model)
    module_to_name = {module: name for name, module in model.named_modules()}
    summary = {}

    for mapping in norm_mappings:
        for norm, *linears in match_modules_set(model, (mapping.norm, *mapping.linears)):
            summary[module_to_name[norm[0]]] = len(tree_leaves(linears))

    assert summary == {
        f"{prefix}.0.input_layernorm": expected_summary["layer0_input"],
        f"{prefix}.0.post_attention_layernorm": expected_summary["layer0_post"],
        f"{prefix}.1.input_layernorm": expected_summary["layer1_input"],
        f"{prefix}.1.post_attention_layernorm": expected_summary["layer1_post"],
        norm_path: expected_summary["final_norm"],
    }
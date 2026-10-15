import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from qwen_tuning.model import MemoryEfficientQwen, chunked_cross_entropy


def tiny_config():
    return Qwen3_5TextConfig(
        vocab_size=41, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=128, tie_word_embeddings=True, use_cache=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 1.0, "mrope_section": [2, 2, 4]},
    )


def test_chunked_loss_and_gradients_match_full_vocabulary():
    torch.manual_seed(3)
    hidden = torch.randn(2, 11, 7, requires_grad=True)
    head = torch.nn.Linear(7, 41, bias=False)
    labels = torch.randint(0, 41, (2, 11))
    labels[0, 3:6] = -100
    expected = F.cross_entropy(head(hidden[:, :-1]).reshape(-1, 41), labels[:, 1:].reshape(-1))
    expected_gradients = torch.autograd.grad(expected, (hidden, head.weight))
    actual = chunked_cross_entropy(hidden, labels, head, chunk_size=3)
    actual_gradients = torch.autograd.grad(actual, (hidden, head.weight))
    torch.testing.assert_close(actual, expected)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_hybrid_qwen_loss_matches_stock_and_adapter_reloads(tmp_path):
    torch.manual_seed(4)
    stock = Qwen3_5ForCausalLM(tiny_config())
    model = MemoryEfficientQwen(tiny_config())
    model.load_state_dict(stock.state_dict())
    model.loss_chunk_size = 3
    ids = torch.randint(0, 41, (1, 12))
    torch.testing.assert_close(model(input_ids=ids, labels=ids).loss, stock(input_ids=ids, labels=ids).loss)
    torch.testing.assert_close(model(input_ids=ids).logits, stock(input_ids=ids).logits)
    stock.save_pretrained(tmp_path / "base")
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4, target_modules="all-linear",
    ))
    modules = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
    assert any("linear_attn.in_proj_qkv" in name for name in modules)
    assert any("self_attn.q_proj" in name for name in modules)
    assert not any("lm_head" in name for name in modules)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01)
    model(input_ids=ids, labels=ids).loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for name, parameter in model.named_parameters() if "lora_B" in name)
    optimizer.step()
    model.save_pretrained(tmp_path / "adapter")
    from peft import PeftModel

    reloaded = PeftModel.from_pretrained(MemoryEfficientQwen.from_pretrained(tmp_path / "base"), tmp_path / "adapter")
    model.eval()
    reloaded.eval()
    torch.testing.assert_close(model(input_ids=ids).logits, reloaded(input_ids=ids).logits)

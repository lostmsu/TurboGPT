"""Text backbone and checkpointed vocabulary chunks for exact causal LM loss."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3_5ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


def chunked_cross_entropy(hidden_states, labels, head, chunk_size: int):
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    hidden = hidden_states[:, :-1].reshape(-1, hidden_states.shape[-1])
    targets = labels[:, 1:].reshape(-1)
    count = (targets != -100).sum().clamp_min(1)

    def loss_chunk(states, target):
        logits = head(states)
        return F.cross_entropy(logits.float(), target, reduction="sum", ignore_index=-100)

    loss = hidden_states.new_zeros((), dtype=torch.float32)
    for start in range(0, targets.numel(), chunk_size):
        states, target = hidden[start:start + chunk_size], targets[start:start + chunk_size]
        # Checkpoint the head AND CE; otherwise autograd retains every chunk's
        # enormous [tokens, 248320] logits/softmax until the backward pass.
        if torch.is_grad_enabled():
            part = checkpoint(loss_chunk, states, target, use_reentrant=False)
        else:
            part = loss_chunk(states, target)
        loss = loss + part
    return loss / count


class MemoryEfficientQwen(Qwen3_5ForCausalLM):
    # Trainer divides the mean loss across accumulation steps. Do not accept
    # num_items_in_batch, which would require a different loss normalization.
    accepts_loss_kwargs = False
    loss_chunk_size = 128

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None,
        past_key_values=None, inputs_embeds=None, labels=None, use_cache=None,
        logits_to_keep=0, **kwargs,
    ):
        if labels is None:
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, use_cache=use_cache,
                logits_to_keep=logits_to_keep, **kwargs,
            )
        kwargs.pop("num_items_in_batch", None)
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, inputs_embeds=inputs_embeds,
            use_cache=False, **kwargs,
        )
        loss = chunked_cross_entropy(outputs.last_hidden_state, labels, self.lm_head, self.loss_chunk_size)
        # Training/evaluation use loss only. Generation follows the stock HF
        # implementation above and still produces ordinary vocabulary logits.
        return CausalLMOutputWithPast(loss=loss)

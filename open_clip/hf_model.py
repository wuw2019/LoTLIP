""" huggingface model adapter

Wraps HuggingFace transformers (https://github.com/huggingface/transformers) models for use as a text tower in CLIP model.
"""
import re
import os
import torch
import torch.nn as nn
from torch import TensorType

try:
    import transformers
    from transformers import AutoModel, AutoTokenizer, AutoConfig, PretrainedConfig
    from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, \
        BaseModelOutputWithPoolingAndCrossAttentions
except ImportError as e:
    transformers = None


    class BaseModelOutput:
        pass


    class PretrainedConfig:
        pass

from .hf_configs import arch_dict


# utils
def _camel2snake(s):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()


# TODO: ?last - for gpt-like models
_POOLERS = {}


def register_pooler(cls):
    """Decorator registering pooler class"""
    _POOLERS[_camel2snake(cls.__name__)] = cls
    return cls


@register_pooler
class MeanPooler(nn.Module):
    """Mean pooling"""

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        masked_output = x.last_hidden_state * attention_mask.unsqueeze(-1)
        return masked_output.sum(dim=1) / attention_mask.sum(-1, keepdim=True)


@register_pooler
class MaxPooler(nn.Module):
    """Max pooling"""

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        masked_output = x.last_hidden_state.masked_fill(attention_mask.unsqueeze(-1), -torch.inf)
        return masked_output.max(1).values


@register_pooler
class ClsPooler(nn.Module):
    """CLS token pooling"""

    def __init__(self, use_pooler_output=True):
        super().__init__()
        self.cls_token_position = 0
        self.use_pooler_output = use_pooler_output

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        if (self.use_pooler_output and
            isinstance(x, (BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions)) and
            (x.pooler_output is not None)
        ):
            return x.pooler_output

        return x.last_hidden_state[:, self.cls_token_position, :]


@register_pooler
class ClsLastHiddenStatePooler(nn.Module):
    """CLS token pooling
    NOTE: this is equivalent to ClsPooler above with use_pooler_output=False
    """

    def __init__(self):
        super().__init__()
        self.cls_token_position = 0

    def forward(self, x: BaseModelOutput, attention_mask: TensorType):
        return x.last_hidden_state[:, self.cls_token_position, :]


class HFTextEncoder(nn.Module):
    """HuggingFace model adapter"""
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            model_name_or_path: str,
            output_dim: int,
            config: PretrainedConfig = None,
            pooler_type: str = None,
            proj_type: str = None,
            pretrained: bool = True,
            output_tokens: bool = False,
            cache_dir: str = None,
            num_corner_token: int = 0
    ):
        super().__init__()
        self.output_tokens = output_tokens
        self.output_dim = output_dim
        if '/' not in model_name_or_path:
            model_name_or_path = os.path.join(cache_dir, model_name_or_path)
        # TODO: find better way to get this information
        uses_transformer_pooler = (pooler_type == "cls_pooler")

        if transformers is None:
            raise RuntimeError("Please `pip install transformers` to use pre-trained HuggingFace models")
        if config is None:
            self.config = AutoConfig.from_pretrained(model_name_or_path)
            create_func, model_args = (AutoModel.from_pretrained, model_name_or_path) if pretrained else (
                AutoModel.from_config, self.config)
            # TODO: do all model configs have this attribute? PretrainedConfig does so yes??
            if hasattr(self.config, "is_encoder_decoder") and self.config.is_encoder_decoder:
                self.transformer = create_func(model_args)
                self.transformer = self.transformer.encoder
            else:
                self.transformer = create_func(model_args, add_pooling_layer=uses_transformer_pooler)
        else:
            self.config = config
            self.transformer = AutoModel.from_config(config)
        if pooler_type is None:  # get default arch pooler
            pooler_type = (arch_dict[self.config.model_type]["pooler"])

        # FIXME downstream users of OpenCLIP models use these attr, need to verify valid across all models
        vocab_size = getattr(self.config, 'vocab_size', 0)
        if num_corner_token > 0:
            vocab_size = vocab_size+num_corner_token
            self.resize_token_embeddings(vocab_size)

        self.vocab_size=vocab_size

        self.context_length = getattr(self.config, 'max_position_embeddings', 0)

        self.pooler = _POOLERS[pooler_type]()

        d_model = getattr(self.config, arch_dict[self.config.model_type]["config_names"]["width"])
        if (d_model == output_dim) and (proj_type is None):  # do we always need a proj?
            self.proj = nn.Identity()
        elif proj_type == 'linear':
            self.proj = nn.Linear(d_model, output_dim, bias=False)
        elif proj_type == 'mlp':
            hidden_size = (d_model + output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(hidden_size, output_dim, bias=False),
            )

        self.num_corner_token = num_corner_token

    def resize_token_embeddings(self, vocab_size):
        num_tokens, embedding_dim = self.transformer.embeddings.word_embeddings.weight.shape

        resized_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # initialize resized word embeddings
        resized_embeddings.weight.data.normal_(mean=0.0, std=self.transformer.config.initializer_range)

        # Setting device and type accordingly
        resized_embeddings.to(
            self.transformer.embeddings.word_embeddings.weight.device,
            dtype=self.transformer.embeddings.word_embeddings.weight.dtype,
        )

        # Copying the old entries
        resized_embeddings.weight.data[:num_tokens, :] = self.transformer.embeddings.word_embeddings.weight.data
        self.transformer.embeddings.word_embeddings = resized_embeddings
    
    def build_attn_mask(self, x):
        attn_mask = (x != self.config.pad_token_id).long()
        if self.num_corner_token == 0:
            return attn_mask
        
        # change the 1d mask to 2d mask [bs, token_num, token_num]
        attn_mask = attn_mask[:, None, :].repeat(1,attn_mask.shape[-1], 1)
        
        for i in range(x.shape[0]):
            cur_input_ids = x[i]
            cur_mask = attn_mask[i]

            cur_mask[:, : self.num_corner_token+1] = cur_mask[:, : self.num_corner_token+1] * 0
            cur_mask[self.num_corner_token+1:, 0] = 1
            for cls_token_id in range(self.num_corner_token+1):
                cur_mask[cls_token_id, cls_token_id] = 1
            attn_mask[i] = cur_mask

        return attn_mask

    def forward(self, x: TensorType):
        
        attn_mask = self.build_attn_mask(x)
        out = self.transformer(input_ids=x, attention_mask=attn_mask)
        
        if (isinstance(out, (BaseModelOutputWithPooling, BaseModelOutputWithPoolingAndCrossAttentions)) and
            (out.pooler_output is not None) and
            (self.num_corner_token > 0)
        ):
            pooled_out = out.last_hidden_state[:,:self.num_corner_token+1,:]
            pooled_out = pooled_out.reshape(-1, pooled_out.shape[-1])
            pooled_out = self.transformer.pooler.dense(pooled_out)
            pooled_out = self.transformer.pooler.activation(pooled_out)
            projected = self.proj(pooled_out)

        else:
            pooled_out = self.pooler(out, attn_mask)
            projected = self.proj(pooled_out)

        seq_len = out.last_hidden_state.shape[1]

        if type(self.pooler) == ClsPooler:
            tokens = out.last_hidden_state[:, torch.arange(seq_len) != self.pooler.cls_token_position, :] 
        else:  tokens = out.last_hidden_state
        
        
        if self.output_tokens:
            return projected, tokens
        return projected

    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        if not unlocked_layers:  # full freezing
            for n, p in self.transformer.named_parameters():
                p.requires_grad = (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False
            return

        encoder = self.transformer.encoder if hasattr(self.transformer, 'encoder') else self.transformer
        layer_list = getattr(encoder, arch_dict[self.config.model_type]["config_names"]["layer_attr"])
        print(f"Unlocking {unlocked_layers}/{len(layer_list) + 1} layers of hf model")
        embeddings = getattr(
            self.transformer, arch_dict[self.config.model_type]["config_names"]["token_embeddings_attr"])
        modules = [embeddings, *layer_list][:-unlocked_layers]
        # freeze layers
        for module in modules:
            for n, p in module.named_parameters():
                p.requires_grad = (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.gradient_checkpointing_enable()

    def init_parameters(self):
        pass

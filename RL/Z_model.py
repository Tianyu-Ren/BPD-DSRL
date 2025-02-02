from transformers import T5EncoderModel, T5Config
from torch import nn
from typing import Optional
import torch


class LogZ(nn.Module):
    def __init__(self, path, classifier_dropout=0.1):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(path)
        self.dropout = nn.Dropout(classifier_dropout)
        self.pooler = nn.Sequential(nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size),
                                    nn.Tanh())
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        encoder_output_logits = self.encoder(input_ids=input_ids,
                                             attention_mask=attention_mask).last_hidden_state
        encoder_output_logits = (encoder_output_logits * attention_mask.unsqueeze(dim=-1)).sum(
            dim=1) / attention_mask.sum(dim=-1, keepdim=True)
        pooled_output = self.pooler(encoder_output_logits)
        pooled_output = self.dropout(pooled_output)
        # pooled_output = encoder_output_logits
        logits = self.classifier(pooled_output)
        return logits

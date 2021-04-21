"""Define model architecture."""

from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor


def create_model(weights):
    """Return a T5 model.
    """
    model = T5ForConditionalGeneration.from_pretrained(weights)
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    optimizer = Adafactor(
        params=model.parameters(),
        lr=1e-4,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False)
    return model, optimizer, tokenizer
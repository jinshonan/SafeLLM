# this is the script used to calculate Perplexity
# It's also the first layer of defense framework

from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import math

def calculate_perplexity(text, model_name='gpt2'):
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model.eval()

    # Tokenize input
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids']

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = math.exp(loss.item())
        return perplexity

# Example usage
text = "The quick brown fox jumps over the lazy dog."
ppl = calculate_perplexity(text)
print(f"Perplexity: {ppl:.2f}")
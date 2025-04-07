from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(model_name):
    print("Loading model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to('cuda' if torch.cuda.is_available() else 'cpu')
        print("Model loaded successfully.")
        return model, tokenizer
    except EnvironmentError as e:
        print(f"Error loading model: {e}")
        print("Ensure the model is downloaded correctly from Hugging Face.")
        exit(1)

def generate_response(model, tokenizer, input_text, max_length=200):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

def chat(model, tokenizer):
    print("Chatbot is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = generate_response(model, tokenizer, user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    MODEL_NAME = "meta-llama/Llama-3.2-1B"  # Replace with the exact model name or path
    
    model, tokenizer = load_model(MODEL_NAME)
    chat(model, tokenizer)
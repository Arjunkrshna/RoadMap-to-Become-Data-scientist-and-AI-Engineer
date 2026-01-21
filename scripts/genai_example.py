#!/usr/bin/env python3
"""
Generative AI Example

This script demonstrates generating text using the Hugging Face transformers library.
We load a small GPT-2 model (e.g., distilgpt2) and generate a continuation of a prompt.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class GenAIExample:
    def __init__(self, model_name: str = "distilgpt2"):
        """
        Initialize the generator with a pre-trained model and tokenizer.

        Parameters
        ----------
        model_name : str
            Name of the pre-trained model from Hugging Face to load.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_text(self, prompt: str, max_length: int = 50) -> str:
        """
        Generate text from the given prompt.

        Parameters
        ----------
        prompt : str
            The initial text to prompt the model.
        max_length : int
            Total length of the generated sequence including the prompt.

        Returns
        -------
        str
            The generated text.
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        # Generate continuation; prevent repeating 2-gram sequences
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text


if __name__ == "__main__":
    # Example usage
    generator = GenAIExample()
    prompt = "Once upon a time,"
    print("Prompt:", prompt)
    generated = generator.generate_text(prompt, max_length=40)
    print("Generated text:")
    print(generated)

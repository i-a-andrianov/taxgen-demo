import os

import torch
from dotenv import load_dotenv
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompting import SYSTEM_PROMPT

load_dotenv()

def load_model_and_tokenizer():
    if torch.cuda.is_available():
        torch.set_default_device(f"cuda:0")
    
    model = AutoModelForCausalLM.from_pretrained(
        'VityaVitalich/TaxoLlama3.1-8b-instruct',
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(
        'meta-llama/Llama-3.1-8B', token=os.getenv("HF_TOKEN")
    )
    
    return model, tokenizer


inference_model, tokenizer = load_model_and_tokenizer()


def predict_node_from_taxollama(word, a, cur_index, last_word=None):
    restop = a.get(word, generate_candidates(word, last_word))
    a[word] = restop
    res = a[word][cur_index]
    return res

def generate_candidates(word, last_word):

    processed_term = f"hypernym: {word}"
    if last_word:
        processed_term += f", hyponym: {last_word}"
    processed_term += " | synset:"

    processed_term = SYSTEM_PROMPT + "\n" + processed_term + "[/INST]"
    input_ids = tokenizer(processed_term, return_tensors="pt")

    gen_conf = {
        "no_repeat_ngram_size": 3,
        "do_sample": True,
        "num_beams": 8,
        "num_return_sequences": 2,
        "max_new_tokens": 32,
        "top_k": 20,
    }

    out = inference_model.generate(inputs=input_ids["input_ids"].to("cuda"), **gen_conf)
    print(tokenizer.batch_decode(out)[0])
    text = tokenizer.batch_decode(out)[0][len(SYSTEM_PROMPT) :].split("[/INST]")[-1]
    print(text)
    return text.split(",")


if __name__ == "__main__":
    print(predict_node_from_taxollama("bichon", {}, 0, ""))

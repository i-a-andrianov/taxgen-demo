import torch
from nltk.corpus import wordnet as wn
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
from collections import defaultdict
from peft import PeftConfig, PeftModel
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print(torch.cuda.is_available())


def predict_node_from_taxollama(word, a, cur_index, last_word=None):
    restop = a.get(word, generate_candidates(word, last_word))
    a[word][cur_index] = restop
    res = a[word][cur_index]
    return res


def generate_candidates(word, last_word):
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
    config = PeftConfig.from_pretrained('VityaVitalich/TaxoLLaMA_All')
    # Do not forget your token for Llama2 models
    if torch.cuda.is_available():
        model = LlamaForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.bfloat16, load_in_4bit=True, token="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")
    else:
        model = LlamaForCausalLM.from_pretrained(config.base_model_name_or_path, token="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") # torch_dtype=torch.bfloat16) #load_in_4bit=True)
    tokenizer = LlamaTokenizer.from_pretrained(config.base_model_name_or_path, token="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")
    inference_model = PeftModel.from_pretrained(model, 'VityaVitalich/TaxoLLaMA_All')

    processed_term = f"hypernym: {word}"
    if last_word:
        processed_term += f", hyponym: {last_word}"
    processed_term += " | synset:"

    system_prompt = """<s>[INST] <<SYS>> You are a helpfull assistant. List all the possible words divided with a coma. Your answer should not include anything except the words divided by a coma<</SYS>>"""
    processed_term = system_prompt + '\n' + processed_term + '[/INST]'

    input_ids = tokenizer(processed_term, return_tensors='pt')

    # This is an example of generation hyperparameters, they could be modified to fit your task
    gen_conf = {
                "no_repeat_ngram_size": 3,
                "do_sample": True,
                "num_beams": 8,
                "num_return_sequences": 2,
                "max_new_tokens": 32,
                "top_k": 20,
            }
    if torch.cuda.is_available():
        out = inference_model.generate(inputs=input_ids['input_ids'].to('cuda'), **gen_conf)
    else:
        out = inference_model.generate(inputs=input_ids['input_ids'], **gen_conf)

    text = tokenizer.batch_decode(out)[0][len(system_prompt):].split('[/INST]')[-1]
    return text.split(',')[0]


if __name__ == '__main__':
    print(predict_node_from_taxollama('bichon', defaultdict(dict), 'addd', ''))

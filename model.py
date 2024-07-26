import together
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline, 
    PretrainedConfig, 
    Qwen2Config, 
    LlamaConfig,
    FalconConfig,
    GemmaConfig,
    Gemma2Config,
    GPT2Config,
    GPTNeoConfig,
    GPTJConfig,
    MistralConfig,
    MixtralConfig,
    PhiConfig,
    Phi3Config,
    Starcoder2Config,
    set_seed
)
import pprint as ppr

set_seed(42)

TOGETHER_API_KEY = '146706bab46d0101232eb1519664fb6e53c5be853f85aae3cfa9abd266e9434d'
together_client = together.Together(api_key=TOGETHER_API_KEY)

base_model_directories = [
    "/data/public_models/huggingface/Qwen/Qwen1.5-0.5B",
    "/data/public_models/huggingface/google/gemma-2b",
    "/data/public_models/huggingface/Qwen/Qwen1.5-1.8B",
    "/data/public_models/huggingface/tiiuae/falcon-7b",
    "/data/public_models/huggingface/deepseek-ai/deepseek-llm-7b-base",
    "/data/public_models/huggingface/meta-llama/Llama-2-7b-hf",
    "/data/public_models/huggingface/Qwen/Qwen1.5-4B",
    "/data/public_models/huggingface/google/gemma-7b",
    "/data/public_models/huggingface/01-ai/Yi-6B",
    "/data/public_models/huggingface/Qwen/Qwen1.5-7B",
    "/data/public_models/huggingface/meta-llama/Llama-2-13b-hf",
    "/data/public_models/huggingface/tiiuae/falcon-40b",
    "/data/public_models/huggingface/mistralai/Mistral-7B-v0.1",
    "/data/public_models/huggingface/01-ai/Yi-9B",
    "/data/public_models/huggingface/meta-llama/Meta-Llama-3-8B",
    "/data/public_models/huggingface/Qwen/Qwen1.5-14B",
    "/data/public_models/huggingface/meta-llama/Llama-2-70b-hf",
    "/data/public_models/huggingface/deepseek-ai/deepseek-llm-67b-base",
    "/data/public_models/huggingface/Qwen/Qwen1.5-32B",
    "/data/public_models/huggingface/Qwen/Qwen1.5-72B",
    "/data/public_models/huggingface/mistralai/Mixtral-8x7B-v0.1",
    "/data/public_models/huggingface/Qwen/Qwen1.5-110B",
    "/data/public_models/huggingface/01-ai/Yi-34B",
    "/data/public_models/huggingface/tiiuae/falcon-180B",
    "/data/public_models/huggingface/meta-llama/Meta-Llama-3-70B",
    "/data/public_models/huggingface/mistralai/Mixtral-8x22B-v0.1",
]

chat_model_directories = [
    "/data/public_models/huggingface/meta-llama/Meta-Llama-3-70B-Instruct",  # Meta llama models, gen 3, 2
    "/data/public_models/huggingface/meta-llama/Meta-Llama-3-8B-Instruct",
    "/data/public_models/huggingface/meta-llama/Llama-2-13b-chat-hf",
    "/data/public_models/huggingface/meta-llama/Llama-2-70b-chat-hf",
    "/data/public_models/huggingface/meta-llama/Llama-2-7b-chat-hf",
    "/data/public_models/huggingface/mistralai/Mistral-7B-Instruct-v0.3",  # Mistral models, gen latest
    "/data/public_models/huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1",
    # "/data/public_models/huggingface/mistralai/Mixtral-8x22B-Instruct-v0.1",        # Skipping for now, tokenizer failing to load
    "/data/public_models/huggingface/Qwen/Qwen1.5-0.5B-Chat",  # Qwen models, gen 2, a few from 1.5
    "/data/public_models/huggingface/Qwen/Qwen1.5-1.8B-Chat",
    "/data/public_models/huggingface/Qwen/Qwen1.5-4B-Chat",
    "/data/public_models/huggingface/Qwen/Qwen2-0.5B-Instruct",
    "/data/public_models/huggingface/Qwen/Qwen2-1.5B-Instruct",
    "/data/public_models/huggingface/Qwen/Qwen2-7B-Instruct",
    "/data/public_models/huggingface/Qwen/Qwen2-72B-Instruct",
    "/data/public_models/huggingface/tiiuae/falcon-7b-instruct",  # Tiiuae (Falcon) models, gen latest # Using hardcoded chat template
    "/data/public_models/huggingface/tiiuae/falcon-40b-instruct",   # Using hardcoded chat template
    "/data/public_models/huggingface/tiiuae/falcon-180B-chat",
    "/data/public_models/huggingface/deepseek-ai/deepseek-llm-7b-chat",  # Deepseek models, gen latest-1
    "/data/public_models/huggingface/deepseek-ai/deepseek-llm-67b-chat",
    "/data/public_models/huggingface/google/gemma-1.1-2b-it",  # Google (Gemma) models, gen 1.1
    "/data/public_models/huggingface/google/gemma-1.1-7b-it",
    "/data/public_models/huggingface/01-ai/Yi-6B-Chat",  # 01-ai (Yi) models, gen 1
    "/data/public_models/huggingface/01-ai/Yi-34B-Chat",
    "/data/sudharsan_sundar/downloaded_models/gemma-2-9b-it",
    "gpt-4o-mini"
]

base_model_name_to_path = {path.split('/')[-1]: path for path in base_model_directories}
chat_model_name_to_path = {path.split('/')[-1]: path for path in chat_model_directories}
all_model_name_to_path = {**base_model_name_to_path, **chat_model_name_to_path}

default_falcon_instruct_chat_template = '''{% if messages[0]['role'] == 'system' %}
    {% set system_message = messages[0]['content'] %}
    {% set messages = messages[1:] %}
{% else %}
    {% set system_message = '' %}
{% endif %}

{{ system_message | trim }}
{% for message in messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}

    {% set content = message['content'].replace('\r\n', '\n').replace('\n\n', '\n') %}
    {{ '\n\n' + message['role'] | capitalize + ': ' + content | trim }}
{% endfor %}

{% if add_generation_prompt %}
    {{ '\n\nAssistant:' }}
{% endif %}'''


class APIModel:
    '''
    Basic wrapper class for getting text responses from api models. 
    Currently only implements together ai api calls, but easily extensible.
    '''
    def __init__(self, model_name, org='together'):
        self.model_name = model_name
        self.org = org

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=60), reraise=True)
    def get_answer_text(self, prompt, system_prompt='You are a helpful assistant.', max_tokens=1024, stop_seqs=None):
        if self.org == 'together':
            if stop_seqs is None:
                response = together_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=max_tokens,
                    top_p=0.0,
                    top_k=1,
                    temperature=0
                )
            else:
                response = together_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=max_tokens,
                    top_p=0.0,
                    top_k=1,
                    temperature=0,
                    stop=stop_seqs
                )

            return response.choices[0].message.content
        else:
            raise NotImplementedError('Not yet configured for other APIs')


class ClusterModel:
    '''
    Basic wrapper class for running inference on models stored on the cluster.
    '''
    def __init__(self, 
                 model_name_or_path,
                 do_sample=False,
                 temperature=0, 
                 top_p=0, 
                 top_k=1, 
                 max_new_tokens=2048,
                 output_scores=False, 
                 batch_size=None):
        if '/' in model_name_or_path:
            self.model_name = model_name_or_path.split('/')[-1]
            self.model_path = model_name_or_path
        else:
            self.model_name = model_name_or_path
            self.model_path = all_model_name_to_path[model_name_or_path]
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                                                       use_fast=False,
                                                       padding_side='left',
                                                       trust_remote_code=True)
        
        if self.model_name in ['falcon-7b-instruct', 'falcon-40b-instruct']:
            self.tokenizer.chat_template = default_falcon_instruct_chat_template
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.pad_token})
        
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.model_path,
                                                          device_map="auto",
                                                          torch_dtype="auto",
                                                          trust_remote_code=True).eval()
        
        # High level class provided by HF for running inference on HF models
        self.inference_pipeline = pipeline(task='text-generation',
                                           model=self.model,
                                           tokenizer=self.tokenizer,
                                           do_sample=do_sample,             # Perform (deterministic) greedy sampling
                                           temperature=None if not do_sample else temperature,
                                           top_p=None if not do_sample else top_p,
                                           top_k=None if not do_sample else top_k,
                                           output_scores=output_scores,      # *Supposed* to also output logits, but *doesn't* seem to work with pipeline obj?
                                           max_new_tokens=max_new_tokens,
                                           return_full_text=False,
                                           batch_size=batch_size)
        
        self.chat_model = 'chat' in model_name_or_path.lower() or 'instruct' in model_name_or_path.lower() or '-it' in model_name_or_path.lower()
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.output_scores = output_scores
        self.max_new_tokens = max_new_tokens
        print('Initialized ClusterModel')

    def get_answer_text(self, prompt, system_prompt=None, max_new_tokens=1024):
        if self.chat_model:
            generation = self.inference_pipeline([{'role': 'user', 'content': prompt}])
        else:
            raise NotImplementedError('Haven\'t set up prompt for base models yet')
        
        return generation[0]['generated_text']
    
    def get_answer_text_batched(self, prompts, system_prompt=None, max_new_tokens=1024):
        old_pad_token_id = self.inference_pipeline.tokenizer.pad_token_id
        self.inference_pipeline.tokenizer.pad_token_id = self.model.config.eos_token_id
        if self.chat_model:
            generations = self.inference_pipeline([[{'role': 'user', 'content': prompt}] for prompt in prompts])
        else:
            raise NotImplementedError('Haven\'t set up prompt for base models yet')
        
        self.inference_pipeline.tokenizer.pad_token_id = old_pad_token_id

        return [generation[0]['generated_text'] for generation in generations]
    
    def get_answer_text_batched_alt(self, prompts, system_prompt=None, max_new_tokens=None):
        '''
        Maximally customizable method for running batched inference on models on the cluster
        '''
        formatted_prompts = prompts
        if self.chat_model:
            formatted_prompts = [self.tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True, tokenize=False) for prompt in prompts]
        
        model_inputs = self.tokenizer(formatted_prompts, return_tensors='pt', padding=True).to('cuda')
        generated_ids = self.model.generate(**model_inputs,
                                            max_new_tokens=self.max_new_tokens if max_new_tokens is None else max_new_tokens,
                                            do_sample=self.do_sample,
                                            temperature=None if not self.do_sample else self.temperature,
                                            top_p=None if not self.do_sample else self.top_p,
                                            top_k=None if not self.do_sample else self.top_k,
                                            output_scores=self.output_scores)       # Can also output logits
        model_outputs = self.tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)

        return model_outputs


def main():
    print('Starting script...')
    # test_problem = '''Consider the following pattern. Each tuple (*) represents (shape type).\nRow 1: (C), (B), (A)\nRow 2: (A), (C), (B)\nRow 3: (B), (A), (?)\n\nPlease determine the correct values for the final tuple of Row 3, (?), which completes the pattern. Please clearly state your final answer as \"The final answer is: [your final answer].\"'''
#     test_problem = '''Consider the following pattern. Each tuple (_) represents (shape type).
# Row 1: (A), (A), (A)
# Row 2: (B), (B), (B)
# Row 3: (C), (C), (?)

# Please determine the correct values for the final tuple of Row 3, (?), which completes the pattern. Please clearly state your final answer as "The final answer is: [your final answer]."'''
#     test_problem2 = '''Consider the following pattern. Each tuple (_) represents (shape type).
# Row 1: (A), (A), (A)
# Row 2: (C), (C), (C)
# Row 3: (B), (B), (?)

# Please determine the correct values for the final tuple of Row 3, (?), which completes the pattern. Please clearly state your final answer as "The final answer is: [your final answer]."'''
#     model_names = ['gemma-2-9b-it']
#     for model_name in model_names:
#         print('MODEL:', model_name)
#         test_model = ClusterModel(model_name)
#         answers = test_model.get_answer_text_batched([test_problem, test_problem2])

#         for answer in answers:
#             print(answer)
#             print('/ / / / /')
#         print('-' * 100)
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-27b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2-27b-it")
    print('MODELS CALLED')


if __name__ == '__main__':
    main()

import together
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, PretrainedConfig, Qwen2Config, LlamaConfig
import pprint as ppr

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
    "/data/public_models/huggingface/mistralai/Mixtral-8x22B-Instruct-v0.1",
    "/data/public_models/huggingface/Qwen/Qwen1.5-0.5B-Chat",  # Qwen models, gen 2, a few from 1.5
    "/data/public_models/huggingface/Qwen/Qwen1.5-1.8B-Chat",
    "/data/public_models/huggingface/Qwen/Qwen1.5-4B-Chat",
    "/data/public_models/huggingface/Qwen/Qwen2-0.5B-Instruct",
    "/data/public_models/huggingface/Qwen/Qwen2-1.5B-Instruct",
    "/data/public_models/huggingface/Qwen/Qwen2-7B-Instruct",
    "/data/public_models/huggingface/Qwen/Qwen2-72B-Instruct",
    "/data/public_models/huggingface/tiiuae/falcon-7b-instruct",  # Tiiuae (Falcon) models, gen latest
    "/data/public_models/huggingface/tiiuae/falcon-40b-instruct",
    "/data/public_models/huggingface/tiiuae/falcon-180B-chat",
    "/data/public_models/huggingface/deepseek-ai/deepseek-llm-7b-chat",  # Deepseek models, gen latest-1
    "/data/public_models/huggingface/deepseek-ai/deepseek-llm-67b-chat",
    "/data/public_models/huggingface/google/gemma-1.1-2b-it",  # Google (Gemma) models, gen 1.1
    "/data/public_models/huggingface/google/gemma-1.1-7b-it",
    "/data/public_models/huggingface/01-ai/Yi-6B-Chat",  # 01-ai (Yi) models, gen 1
    "/data/public_models/huggingface/01-ai/Yi-34B-Chat"
]

base_model_name_to_path = {path.split('/')[-1]: path for path in base_model_directories}
chat_model_name_to_path = {path.split('/')[-1]: path for path in chat_model_directories}
all_model_name_to_path = {**base_model_name_to_path, **chat_model_name_to_path}

# TODO: Need to map model family to Config type
# TODO: Need to map model family to instruct prompt format


class APIModel:
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
    def __init__(self, model_name_or_path, temperature=0, top_p=0, top_k=1, max_new_tokens=1024, output_scores=False):
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
        self.config = LlamaConfig().from_pretrained(self.model_path)            # TODO: Automatically choose the right Config object
        self.config.do_sample = False   # Sets generation to greedy decoding, i.e. temp=0 i.e. top_p=0, i.e. top_k=1
        self.config.max_new_tokens = max_new_tokens
        self.config.output_scores = output_scores  # Set to True if you want model generations to output logits as well
        print(self.config)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=self.model_path,
                                                          device_map="auto",
                                                          torch_dtype="auto",
                                                          trust_remote_code=True,
                                                          config=self.config).eval()
        self.inference_pipeline = pipeline(task='text-generation',
                                           model=self.model,
                                           tokenizer=self.tokenizer,
                                           config=self.config,
                                           do_sample=False,
                                           temperature=None,
                                           top_p=None,
                                           top_k=None)
        print('Initialized ClusterModel')

    def get_answer_text(self, prompt, system_prompt=None, max_new_tokens=1024):
        # TODO: Automatically format the prompt in instruction format based on the model family
        # TODO: Consider batching if input length is highly regular to increase throughput
        # TODO: Look back at Gene's implementation to see how mine compares
        generation = self.inference_pipeline(prompt)
        print('GENERATION\n', generation)

        return generation[0]['generated_text'][len(prompt):]


def main():
    print('Starting script...')
    test_model = ClusterModel('Meta-Llama-3-8B-Instruct')
    '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>

What is France's capital?<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
    print(test_model.get_answer_text(prompt='<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nConsider the following pattern. Each tuple (*) represents (shape type).\nRow 1: (C), (B), (A)\nRow 2: (A), (C), (B)\nRow 3: (B), (A), (?)\n\nPlease determine the correct values for the final tuple of Row 3, (?), which completes the pattern. Please clearly state your final answer as \"The final answer is: [your final answer].\"<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>'))

    # TODO: Check to make sure that generation works as intended
    # TODO: Make sure parsing pipeline works as intended
    # TODO: Do small test and make sure eval works right end to end


if __name__ == '__main__':
    main()

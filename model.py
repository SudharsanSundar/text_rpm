import together
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, PretrainedConfig
import pprint as ppr

TOGETHER_API_KEY = '146706bab46d0101232eb1519664fb6e53c5be853f85aae3cfa9abd266e9434d'
together_client = together.Together(api_key=TOGETHER_API_KEY)

model_directories = [
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

model_name_to_path = {path.split('/')[-1]: path for path in model_directories}


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
    def __init__(self, model_name, temperature=0, top_p=0, top_k=1, max_len=1024, output_scores=False):
        self.model_name = model_name
        self.model_path = model_name_to_path[model_name]
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                                          device_map="auto",
                                                          torch_dtype="auto",
                                                          trust_remote_code=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path,
                                                       use_fast=False,
                                                       padding_side='left',
                                                       trust_remote_code=True)
        self.config = PretrainedConfig().from_pretrained(self.model_path)
        self.config.temperature = temperature
        self.config.top_p = top_p
        self.config.top_k = top_k
        self.max_length = max_len
        self.config.output_scores = output_scores      # Set to True if you want model generations to output logits as well
        self.inference_pipeline = pipeline(task='text-generation',
                                           model=self.model,
                                           tokenizer=self.tokenizer,
                                           config=self.config)

    def get_answer_text(self, prompt, system_prompt=None, max_tokens=1024):
        generation = self.inference_pipeline(prompt)

        return generation[0]['generated_text']


def main():
    print()
    # test_model = ClusterModel('Qwen1.5-0.5B')

    # TODO: Check to make sure that generation works as intended
    # TODO: Make sure parsing pipeline works as intended
    # TODO: Do small test and make sure eval works right end to end


if __name__ == '__main__':
    main()

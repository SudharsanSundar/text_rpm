import openai
import json
import pprint as ppr

client = openai.OpenAI()


def generate_oai_batch_eval_file(eval_problems_fp, save_fp, model_name='gpt-4o-mini'):
    assert '.jsonl' in save_fp

    problems = []
    with open(eval_problems_fp, 'r') as f:
        for item in f:
            problems.append(json.loads(item))

    idx = 0
    problem_dicts = []
    for problem in problems:
        api_call_dict = {
            'custom_id': str(idx),
            'method': "POST",
            'url': '/v1/chat/completions',
            'body': {
                'model': model_name,
                'messages': [
                    {'role': 'user', 'content': problem['problem_prompt']}
                ],
                'max_tokens': 2048,
                'temperature': 0.0,
                'seed': 42
            },
        }
        problem_dicts.append(api_call_dict)

        idx += 1
    
    with open(save_fp, 'w') as f:
        for item in problem_dicts:
            f.write(json.dumps(item) + '\n')

    print('CREATED BATCH JOB DATA.')


def submit_batch_job(batch_fp):
    batch_input_file = client.files.create(
        file=open(batch_fp, 'rb'),
        purpose='batch'
    )

    batch_input_file_id = batch_input_file.id
    response = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint='/v1/chat/completions',
        completion_window='24h',
        metadata={
            'description': 'eval gpt4o-mini on text rpm v2',
            'data_fp': batch_fp
        }
    )

    print('BATCH JOB SUBMITTED:')
    print(response)


def check_on_batch_jobs():
    print('CURRENT BATCH JOBS:')
    print(client.batches.list(limit=10))


def retrieve_and_save_job_result(output_file_id, eval_problems_fp, save_fp):
    file_response = client.files.content(output_file_id).text
    output_sorted = [None] * len(file_response)
    
    # TODO: Map batch results to eval problems, 
    # TODO: create list holding them in the desired format
    
    with open(save_fp, 'w') as f:
        for item in output_sorted:
            f.write(json.dumps(item) + '\n')


def main():
    # Reference pg: https://platform.openai.com/docs/guides/batch/getting-started
    eval_problems_fp = './datasets/default_rpm_dataset_eval_problems_v2.json'
    batch_fp = './datasets/default_rpm_dataset_eval_problems_v2_oai_batch.jsonl'
    output_file_id = 'file-fmCLFblBKykGMBm9HXrHWnBh'

    # TODO: Make sure save_fp is correct and can work with existing eval_analysis script
    output_save_fp = './v2_results/rpm_eval_results_gpt40-mini.json'

    # generate_oai_batch_eval_file(
    #     eval_problems_fp=eval_problems_fp,
    #     save_fp=batch_fp
    # )

    # submit_batch_job(batch_fp)

    # check_on_batch_jobs()

    retrieve_and_save_job_result(output_file_id, output_save_fp)


if __name__ == '__main__':
    main()

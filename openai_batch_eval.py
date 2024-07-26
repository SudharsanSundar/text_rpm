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


def extract_answer(model_answer):
    '''
    Model answer parsing utility
    '''
    if model_answer.lower().rfind('the final answer is: ') != -1:
        first_cut = model_answer[model_answer.rfind('final answer is: ') + len('final answer is: '):]
        second_cut = first_cut.split(')')[0]
        extracted_answer = second_cut + ')'

        return extracted_answer
    else:
        backup_pattern = r'\([A-Za-z, ]+\)'     # WARNING: assumes values are capital or lowercase letters A-Z, a-z
        backup_find = list(re.findall(backup_pattern, model_answer))
        if len(backup_find) > 0:
            return backup_find[-1]
        else:
            return 'Could not parse answer.'


def retrieve_and_save_job_result(eval_problems_fp, output_file_id, save_fp):
    file_response = client.files.content(output_file_id).text
    batch_eval_results = []
    for line in file_response.split('\n'):
        if len(line) > 0:
            batch_eval_results.append(json.loads(line))

    eval_problems = []
    with open(eval_problems_fp, 'r') as f:
        for item in f:
            eval_problems.append(json.loads(item))
    
    output_sorted = [None] * len(batch_eval_results)
    
    total_score = 0
    for item in batch_eval_results:
        idx = int(item['custom_id'])
        problem = eval_problems[idx]

        prompt = problem['problem_prompt']
        model_answer = item['response']['body']['choices'][0]['message']['content']
        extracted_answer = extract_answer(model_answer)
        correct_answer = problem['problem_answer']

        result = {
            'problem_prompt': prompt,
            'correct_answer': correct_answer,
            'model_answer': model_answer,
            'extracted_answer': extracted_answer,
            'score': int(extracted_answer == correct_answer),
            'problem_characteristics': problem['characteristics']
        }
        output_sorted[idx] = result

        total_score += int(extracted_answer == correct_answer)
    
    with open(save_fp, 'w') as f:
        for item in output_sorted:
            f.write(json.dumps(item) + '\n')
    
    ppr.pprint(output_sorted[:2])
    ppr.pprint(output_sorted[-2:])
    print(f'TOTAL SCORE: {total_score} / {len(batch_eval_results)} = {total_score / len(batch_eval_results)}')


def main():
    # Reference pg: https://platform.openai.com/docs/guides/batch/getting-started
    eval_problems_fp = './datasets/default_rpm_dataset_eval_problems_v2.json'
    batch_fp = './datasets/default_rpm_dataset_eval_problems_v2_oai_batch.jsonl'
    output_file_id = 'file-fmCLFblBKykGMBm9HXrHWnBh'
    output_save_fp = './v2_results/rpm_eval_results_gpt-4o-mini.json'

    # generate_oai_batch_eval_file(
    #     eval_problems_fp=eval_problems_fp,
    #     save_fp=batch_fp
    # )

    # submit_batch_job(batch_fp)

    # check_on_batch_jobs()

    # retrieve_and_save_job_result(eval_problems_fp, output_file_id, output_save_fp)


if __name__ == '__main__':
    main()

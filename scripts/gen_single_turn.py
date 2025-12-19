import torch
import pandas as pd
from RL.setting import *
from RL.gen_util import *
from RL.Dialog_generator import *
from tqdm import tqdm
import os
import sys
import argparse
tqdm.pandas()


def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Generate single-turn responses based on the fixed prompts')

    parser.add_argument('--prompt_path', type=str, default=f'{ROOT}/dataset/eval/gen/seq2seq/single_turn/gen_few/CHI.csv',
                        help='Path to the prompt file; the name should be the same as the tested role')

    parser.add_argument('--gen_path', type=str, default=f'{ROOT}/dataset/eval/gen/seq2seq/single_turn/gen_cot',
                        help='Output Path to the generation')

    parser.add_argument('--ref_path', type=str, default=f'{ROOT}/dataset/eval/gen/seq2seq/file_10.csv',
                        help='Reference Path to the dialogue names')

    parser.add_argument('--demo_path', type=str, default=f'{ROOT}/dataset/eval/gen/seq2seq/multi_turn/CHI.csv',
                        help='Output Path to the generation')

    parser.add_argument('--dem_turn', type=int, default=0,
                        help='few-shot prompting; the number of turns as input')

    parser.add_argument('--cot', action="store_true",
                        help='whether to apply chain-of-thought reasoning')

    parser.add_argument('--test_model_lst', type=list, default=['llama3','mistral','qwen2.5'],
                        help='the role to be tested')  # ['blenderbot','finetuned','llama3',"chatgpt-4o-latest",'mistral','qwen2.5']

    parser.add_argument('--finetune_path', type=str, default='',
                        help='Path to the finetuned model; only add this when we have finetuned the model WITHOUT Lora; \
                         e.g.f{ROOT}/model/seq2seq/merged_no_lora_CHI')

    parser.add_argument('--lora_path', default='',
                        help='Path to the lora layer of the fine-tuned model; only add this when using Lora \
                             e.g. f{ROOT}/model/seq2seq/merged_lora_ADULT')

    parser.add_argument('--month_lst', default=[24,36,48,60],
                        help='the tested age')

    parser.add_argument('--dialogue_num', default=10,
                        help='the number of dialogues to generate')

    parser.add_argument('--max_length', default=15,
                        help='max number of tokens in one sent')

    parser.add_argument('--special_token', default='',
                        help='a list of special tokens added to the tokenizer; add <PAUSE> for lora path')

    parser.add_argument("--debug", action="store_true",
                        help="if debug, generate first 4 rows")

    return parser.parse_args(argv)




def generate(initial_input, generator,model_name, system_role,max_token=50,sil_prob=0) -> list:
    '''buiild single dialogue based on intial prompt'''

    # Determine if we generate <SILENCE> based on the probability `silence_prob`
    if random.random() < sil_prob:
        gen = '<SILENCE>'
    else:
        # preprocess the initial input
        if model_name in ['blenderbot','DialoGPT','finetuned']:
            gen = generator.generate_ref(initial_input, max_length=max_token)

        else:
            messages = [
                {'role': 'system', 'content': system_role},
                {'role': 'user', 'content': initial_input}
            ]
            if model_name in gpt_lst:
                gen = generator.generate_gpt(messages)

            elif model_name in LLM_lst:
                # store history turns for the adult input
                gen, _ = generator.generate_LLM(messages)


    print(system_role)
    print(f'Input: {initial_input}')
    print(f'Gen: {gen}')
    print('--------------------------------------')
    return gen


# TODO: get the function out of the main loop


def main(argv):
    # Args parser
    args = parseArgs(argv)

    # load path args
    prompt_path = args.prompt_path
    ref_path = args.ref_path
    gen_path = args.gen_path
    # Create the parent directory if it does not exist
    os.makedirs(gen_path, exist_ok=True)

    # load device and model args
    device = torch.device("mps" if torch.cuda.is_available() else "cpu") # Check if a GPU is available, otherwise use the CPU
    test_model_lst = args.test_model_lst
    finetune_path = args.finetune_path
    special_token = args.special_token
    lora_path = args.lora_path
    model_lst = load_para_dict(model_para_lst,'model_name',test_model_lst)

    # load generation args
    max_length = args.max_length
    month_lst = args.month_lst
    dialogue_num = args.dialogue_num
    test_role = prompt_path.split('/')[-1].split('.')[0].split('_')[0]
    prompt_role = role_dict[test_role]

    print('######################################### loading prompt data')
    dem_all = pd.read_csv(args.demo_path)
    data = pd.read_csv(prompt_path)
    if len(ref_path) > 0:
        if ref_path.endswith('csv'):
            ref_data = pd.read_csv(ref_path)
        elif ref_path.endswith('xlsx'):
            ref_data = pd.read_excel(ref_path)

    # loop over different age
    gen_all = pd.DataFrame()
    for month in tqdm(month_lst):
        dem = dem_all[dem_all['month']==month]
        # select the given month
        gen_data = data[data['month'] == month]
        # sort data to make sure you have the same order
        dem = dem.sort_values('path')
        gen_data = gen_data.sort_values('path')

        if len(ref_path) >0:
            if dialogue_num != 'all':
                gen_data = gen_data[gen_data['path'].isin(list(set(ref_data[ref_data['month']==month]['path']))[:dialogue_num])]
            else:
                gen_data = gen_data[gen_data['path'].isin(list(set(ref_data[ref_data['month'] == month]['path'])))]
            print(f'File has been loaded from: {ref_path}')
        else:
            gen_data = gen_data[gen_data['path'].isin(list(set(gen_data['path']))[:dialogue_num])]
        print(f'Loading {str(dialogue_num)} dialogue by {str(month)} month children')

        # Group data by dialogue
        dialogue_groups = gen_data[gen_data['month'] == month].groupby('path')
        path_lst = list(set((dem['path'])))

        # loop over the different dialogues
        for dialogue_idx, (dialogue_path, dialogue_group) in enumerate(dialogue_groups):

            if dialogue_idx < len(path_lst)-1:
                dem_dialog = dem[dem['path']==path_lst[dialogue_idx+1]]
            else:
                dem_dialog = dem[dem['path'] == path_lst[0]]   # for the last file in the list
            system_role = get_prompt(test_role, month, include_len=True, dem = dem_dialog, speaker_header = 'speaker',
                            content_header = 'CHILDES', dem_turns = args.dem_turn,gen_length=len_dict[test_role],cot=args.cot)

            if args.debug:
                dialogue_group = dialogue_group.head(1)
                print('#####################################')
                print('Entering the debugging mode!!!')

            for para in model_lst:
                model_name = para['model_name']
                print(model_name)
                try:
                    print(f'Generating from {model_name}')
                    if model_name in nonLLM_lst:
                        if model_name == 'finetuned':
                            print('Applying the fine-tuned model!!!')
                            if len(finetune_path) > 0:
                                para['model_path'] = finetune_path
                            elif len(lora_path) > 0:
                                para['model_path'] = lora_path
                            tokenizer, model = load_model(para['model_name'], para['tokenizer_name'],
                                                      para['model_path'],device, special_token, lora_path)
                        else:
                            tokenizer, model = load_model(para['model_name'], para['tokenizer_name'],
                                                          para['model_path'], device)

                        print('Tokenizer and model have been loaded')
                    else:
                        tokenizer, model = '',''

                    # Create an instance of the TextGenerator class
                    generator = TextGenerator(model_name, tokenizer, model, device)
                    # set sil_prob as 0 as we do not want to explicitly control the prop of the silent turns
                    dialogue_group[model_name] = dialogue_group[prompt_role].progress_apply(lambda text: generate(text, generator,
                                                    model_name,system_role,max_length))

                    gen_all = pd.concat([gen_all,dialogue_group])

                except:
                    # save the results if there is any intermediate generations
                    print('Something wrong with generation initialization!')
                    if gen_all.shape[0] > 0:
                        gen_all.to_csv(f'log_{str(month)}_{model_name}.csv')
                    print(f'Saving the intermediate generations to {gen_path}/log.csv')


        # save the results
        if args.debug:
            filename = f'{test_role}_{dialogue_num}_debug.csv'
        else:
            filename = f'{test_role}_{dialogue_num}.csv'

        # remove blank rows in the file
        gen_all.to_csv(f'{gen_path}/{filename}')
        print('##########################')
        print('Finished generation!!!!')
        print(f'Save the generation to {gen_path}/{filename}')
        print('##########################')



if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)

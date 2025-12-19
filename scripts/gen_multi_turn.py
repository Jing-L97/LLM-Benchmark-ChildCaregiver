import argparse
from tqdm import tqdm
import os
from RL.setting import *
from RL.gen_util import *
from RL.Dialog_generator import *
import torch
import sys

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Generate multi-turn interactions between reference and test models')

    parser.add_argument('--gen_path', type=str, default=f'{ROOT}/dataset/eval/gen/seq2seq/multi_turn/CHI.csv',
                        help='Path to the generated texts')

    parser.add_argument('--finetune_path', type=str, default= '',
                        help='Path to the finetuned model; only add this when we have finetuned the model WITHOUT Lora; \
                             e.g.f={ROOT}/model/seq2seq/merged_no_lora_CHI')

    parser.add_argument('--ref_path', type=str, default=f'{ROOT}/dataset/eval/gen/seq2seq/file_10.csv',
                        help='Reference Path to the dialogue names')

    parser.add_argument('--demo_path', type=str, default=f'{ROOT}/dataset/eval/gen/seq2seq/multi_turn/CHI.csv',
                        help='Output Path to the generation')

    parser.add_argument('--dem_turn', type=int, default=6,
                        help='few-shot prompting; the number of turns as input')

    parser.add_argument('--lora_path', default='',
                        help='Path to the lora layer of the fine-tuned model; only add this when using Lora')

    parser.add_argument('--out_path', type=str,default=f"{ROOT}/dataset/eval/gen/seq2seq/multi_turn/gen",
                        help='Path to the generated results')

    parser.add_argument('--test_role', type=str, default='ADULT',
                        help='the role to be tested: CHI or ADULT')

    parser.add_argument('--test_model_lst', type=list, default= ['chatgpt-4o-latest'],
                        help='model names to be tested e.g.chatgpt-4o-latest')

    parser.add_argument('--ref_model_name', type=str, default='chatgpt-4o-latest',
                        help='the ref model name: finetuned or llama3')

    parser.add_argument('--ref_history_turns', default=4,
                        help='the turn history for the child production')

    parser.add_argument('--test_history_turns', default=4,
                        help='the turn history for the adult production')

    parser.add_argument('--month_lst', default=[24,36,48,60],
                        help='the tested age')

    parser.add_argument('--dialogue_num', default=10,
                        help='the number of dialogues to generate; also set all as in the ref data')

    parser.add_argument('--special_token', default='',
                        help='the content of the special token: either nothing or <SILENCE>')

    parser.add_argument('--debug', default=False,
                        help='whether to debug the code with 10 rows')

    return parser.parse_args(argv)


def cut_dataframe1(df, column_name, start_value):
    """
    Cut off rows from a DataFrame until a given column contains the specified start value.
    Returns a DataFrame starting from the first occurrence of start_value in column_name.

    Parameters:
    df (pd.DataFrame): The DataFrame to be sliced.
    column_name (str): The column to check for the start_value.
    start_value: The value in column_name to start from.

    Returns:
    pd.DataFrame: The sliced DataFrame starting from start_value.
    """
    try:
        # Find the index of the first occurrence of start_value in the specified column
        start_idx = df[df[column_name] == start_value].index[0]
        # Slice the DataFrame from the start_idx
        return df.loc[start_idx:].reset_index(drop=True)
    except IndexError:
        print(f"Value {start_value} not found in column '{column_name}'.")
        return df  # Return the original DataFrame if start_value is not found


def cut_dataframe(df, column_name, start_value, reset_index=True):
    """
    Cut off rows from a DataFrame until a given column contains the specified start value.
    Returns a DataFrame starting from the first occurrence of start_value in column_name.

    Parameters:
    df (pd.DataFrame): The DataFrame to be sliced.
    column_name (str): The column to check for the start_value.
    start_value: The value in column_name to start from.
    reset_index (bool): Whether to reset the index after slicing. Default is True.

    Returns:
    pd.DataFrame: The sliced DataFrame starting from start_value.
                 Returns empty DataFrame if column_name doesn't exist.
                 Returns original DataFrame if start_value not found.

    Raises:
    KeyError: If column_name doesn't exist in the DataFrame.
    """
    # Input validation
    if df is None or df.empty:
        return pd.DataFrame()

    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame")

    # Create a copy with reindexed continuous integers
    df_reindexed = df.reset_index(drop=True)

    # Find matching rows in reindexed DataFrame
    matching_rows = df_reindexed[df_reindexed[column_name] == start_value]

    if matching_rows.empty:
        print(f"Warning: Value '{start_value}' not found in column '{column_name}'.")
        return df_reindexed if reset_index else df

    # Get the first matching index from reindexed DataFrame
    start_idx = matching_rows.index[0]

    # Slice the reindexed DataFrame
    result_df = df_reindexed.loc[start_idx:]

    # Only reset index if specifically requested, since we already have clean indices
    if not reset_index:
        # Restore original indexing by mapping back to original DataFrame
        result_df.index = df.index[start_idx:]

    return result_df


def load_para(test_role,test_para,ref_para,ref_history_turns,test_history_turns,device):

    child_para = {
        'device': device,
        'header_child': 'CHI',
        'max_token': 10,
        'history_turns': ref_history_turns
    }

    adult_para = {'device': device,
                  'header_adult': 'ADULT',
                  'max_token': 12,
                  'history_turns': test_history_turns}

    if test_role == 'CHI':
        child_para.update(test_para)
        adult_para.update(ref_para)

    elif test_role == 'ADULT':
        child_para.update(ref_para)
        adult_para.update(test_para)

    return child_para, adult_para




def gen_pipeline(data:pd.DataFrame,dem:pd.DataFrame, child_para:dict, adult_para:dict, sil_dict:dict, len_dict:dict,
                 device,nonLLM_lst:list, special_token:str,lora_path:str,output_path:str,month:int,dem_turn:int,debug=False):
    # create the output folder if there does not exsit
    os.makedirs(output_path, exist_ok=True)
    # load model and tokenizer
    para = child_para
    if para['model_name'] in nonLLM_lst:
        child_tokenizer, child_model = load_model(para['model_name'], para['tokenizer_name'], para['model_path'],
                                              para['device'], special_token,lora_path)
        print('Finish loading tokenizers')
    else:    # add place holder in the simulation
        child_tokenizer, child_model = '',''
        print('Skip loading tokenizer and model for child role')
    print('Reference model and tokenizer loaded!')

    para = adult_para
    if para['model_name'] in nonLLM_lst:
        adult_tokenizer, adult_model = load_model(para['model_name'], para['tokenizer_name'], para['model_path'],
                                              para['device'], special_token,lora_path)
    else:    # add place holder in the simulation
        adult_tokenizer, adult_model = '',''
        print('Skip loading tokenizer and model for adult role')

    print('Test model and tokenizer loaded!')
    print('#####################################')
    print('Generating new conversation!')
    # loop over the same conversation

    log_lst = []
    child_model_name = child_para['model_name']
    adult_model_name = adult_para['model_name']

    dialogue_groups = data.groupby('path')
    path_lst = list(set((dem['path'])))
    gen_all = pd.DataFrame()
    # loop over the different dialogues
    for dialogue_idx, (dialogue_path, dialogue_group) in enumerate(dialogue_groups):
        try:
            if dialogue_idx < len(path_lst) - 1:
                dem_dialog = dem[dem['path'] == path_lst[dialogue_idx + 1]]
            else:
                dem_dialog = dem[dem['path'] == path_lst[0]]  # for the last file in the list
            child_prompt = get_prompt('CHI', month, include_len=True, dem=dem_dialog, speaker_header='speaker',
                                     content_header='CHILDES', dem_turns=dem_turn, gen_length=len_dict['CHI'])
            adult_prompt = get_prompt('ADULT', month, include_len=True, dem=dem_dialog, speaker_header='speaker',
                                      content_header='CHILDES', dem_turns=dem_turn, gen_length=len_dict['ADULT'])

            # cut the df so that it starts fromt eh child turn
            data_group = cut_dataframe(dialogue_group, 'speaker', 'CHI')
            initial_input = data_group[data_group['speaker']=='CHI'].head(1)['CHILDES'].item()

            if debug:
                data_group = data_group.head(4)
                print('#####################################')
                print('Entering the debugging mode!!!')

            num_turns = int(data_group.shape[0] / 2)

            simulator = DialogueSimulator(child_model_name,  adult_model_name,
                                                    child_tokenizer, child_model,adult_tokenizer, adult_model,
                                                    sil_dict,device,child_prompt,adult_prompt,max_child_token=child_para['max_token'],
                                                    max_adult_token=adult_para['max_token'],
                                                    child_history_turns=child_para['history_turns'],
                                                    adult_history_turns=adult_para['history_turns'], num_turns=num_turns)
            conversation_log = simulator.simulate_conversation(initial_input)
            # save the seperate files into a given folder in case any bug
            data_group['gen'] = conversation_log[:data_group.shape[0]]
            # replace the speaker into correct turns
            speaker_order = ['CHI', 'ADULT']
            data_group['speaker'] = [speaker_order[i % 2] for i in range(data_group.shape[0])]

            data_group.rename(columns={'gen': f'{child_model_name}_{adult_model_name}'}, inplace=True)
            filename = dialogue_path.split('/')[-1]
            data_group.to_csv(f'{output_path}/{filename}.csv')
            gen_all = pd.concat([gen_all,data_group])

        except:
            log_lst.append(dialogue_path)

    return gen_all,log_lst




def main(argv):
    # Args parser
    args = parseArgs(argv)

    # load generation args
    ref_history_turns = args.ref_history_turns
    test_history_turns = args.test_history_turns
    test_role = args.test_role
    dialogue_role = role_dict[test_role]
    dialogue_num = args.dialogue_num

    # load path args
    gen_path = args.gen_path
    ref_path = args.ref_path
    finetune_path = args.finetune_path
    out_path = args.out_path
    # create parent folder if the path does not exist
    out_path = f'{out_path}/{test_role}'
    if test_role == 'CHI':
        filename = f'CHI_{str(test_history_turns)}_ADULT_{str(ref_history_turns)}_{str(dialogue_num)}'
    elif test_role == 'ADULT':
        filename = f'CHI_{str(ref_history_turns)}_ADULT_{str(test_history_turns)}_{str(dialogue_num)}'


    # load device and model args
    test_model_lst = args.test_model_lst
    ref_model_name = args.ref_model_name
    special_token = args.special_token
    lora_path = args.lora_path
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    model_lst = load_para_dict(model_para_lst,'model_name',test_model_lst)
    ref_para = load_para_dict(model_para_lst,'model_name',[ref_model_name])
    if ref_model_name == 'finetuned':
        print('Applying the fine-tuned model!!!')
        if len(finetune_path) > 0:
            ref_para[0]['model_path'] = finetune_path
            # check role
            check_role(finetune_path, dialogue_role)
        elif len(lora_path) > 0:
            ref_para[0]['model_path'] = lora_path
            check_role(lora_path, dialogue_role)

    sil_dict.update({test_role: 0})

    # load transcript
    data_all = pd.read_csv(gen_path)
    if ref_path.endswith('csv'):
        ref_data = pd.read_csv(ref_path)
    elif ref_path.endswith('xlsx'):
        ref_data = pd.read_excel(ref_path)
    dem_all = pd.read_csv(args.demo_path)


    # loop over different month list
    gen_all = pd.DataFrame()
    log_all = pd.DataFrame()
    for month in tqdm(args.month_lst):
        # select the given month
        gen_data = data_all[data_all['month'] == month]
        dem = dem_all[dem_all['month'] == month]
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
            # select the corresponding file
            gen_data = gen_data[gen_data['path'].isin(list(set(gen_data['path']))[:dialogue_num])]


        # compute the ref role sil proportion
        sil_prop = gen_data[(gen_data['speaker']==dialogue_role)&
                    (gen_data['CHILDES'].isin(['<EMPTY>','<NONSPEECH>', '<UNINTELLIGIBLE>']))].shape[0]/gen_data.shape[0]
        sil_dict.update({dialogue_role:sil_prop})

        # loop over the model list
        for model_para in tqdm(model_lst):
            # load para
            model_name = model_para['model_name']
            # create the ouput dir
            ref_name = ref_model_name.split('-')[0]
            test_name = model_name.split('-')[0]
            gen_model_path = f'{out_path}/{ref_name}_{test_name}'
            # Create the parent directory if it does not exist
            os.makedirs(gen_model_path, exist_ok=True)

            print('#####################################')
            print(f'Testing model {model_name}')
            output_path = f'{gen_model_path}/files_CHI_{str(ref_history_turns)}_ADULT_{str(test_history_turns)}'
            child_para, adult_para = load_para(test_role,model_para,ref_para[0],ref_history_turns,test_history_turns,device)
            data,log_lst = gen_pipeline(gen_data, dem,child_para, adult_para, sil_dict, len_dict,device,nonLLM_lst,
                                    special_token,lora_path,output_path =output_path,month=month,dem_turn=args.dem_turn,debug=args.debug)

            # rename the gen header
            data.rename(columns={'gen': f'{ref_model_name}_{model_name}'}, inplace=True)
            gen_all = pd.concat([gen_all, data])
            log_frame = pd.DataFrame(log_lst).T
            log_all = pd.concat([log_all, log_frame])

        gen_all.to_csv(f'{gen_model_path}/{filename}.csv')
        log_all.to_csv(f'{gen_model_path}/log_{filename}.csv')
        print(f'Have saved the file to:{gen_model_path}/{filename}.csv')

    print(log_all)




if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)

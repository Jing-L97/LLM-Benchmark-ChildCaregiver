from transformers import AutoModelForCausalLM, AutoTokenizer,BlenderbotTokenizer, BlenderbotForConditionalGeneration
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
import pandas as pd
from RL.setting import *
from dotenv import load_dotenv
load_dotenv()
from RL.setting import *


##########################
# func to preprocess data
##########################

def remove_col(data, start_header):
    # Get the index of the starting column header
    col_idx = data.columns.get_loc(start_header)
    # Slice the DataFrame to include the given column and all columns to its right
    df_slice = data.iloc[:, col_idx:]
    return df_slice

def load_transcript(data, header_lst, upper_header, lower_header):

    def get_speaker(text):
        return text[:-4]
    # List to store the new rows
    new_rows = []

    # Iterate row by row
    for index, row in data.iterrows():
        # Collect columns to duplicate
        duplicate_cols = {col: row[col] for col in header_lst}

        # Append row with upper_header (A) value
        new_rows.append({
            'CHILDES': row[upper_header],
            'Speaker': upper_header,
            **duplicate_cols
        })

        # Append row with lower_header (B) value
        new_rows.append({
            'CHILDES': row[lower_header],
            'Speaker': lower_header,
            **duplicate_cols
        })

    # Create a new DataFrame from the new_rows list
    df_merged = pd.DataFrame(new_rows)
    # replace the speaker labels
    df_merged['Speaker'] = df_merged['Speaker'].apply(get_speaker)
    return df_merged



def match_info(freq,concret):
    concret = concret[['Word', 'Conc.M']]
    concret.rename(columns={'Word': 'word','Conc.M':'conc'}, inplace=True)
    # Merge with how='left' to keep all rows from df1
    merged_df = pd.merge(freq, concret, on='word', how='left')
    # Replace NaN values in colB with 0
    merged_df['conc'] = merged_df['conc'].fillna(0)
    return merged_df

    
####################################
# func to load tokenizer and model
####################################
def load_para_dict(dict_list:list,key:str,values:list)->list:
    selected_dicts = [d for d in dict_list if d.get(key) in values]
    return selected_dicts

def add_special_token(special_token:str,tokenizer,model):

    # Add the special token to the tokenizer
    tokenizer.add_tokens([special_token])
    # Resize the model's token embeddings to accommodate the new token
    model.resize_token_embeddings(len(tokenizer))
    # Verify that the special token has been added
    print(f"Special token ID: {tokenizer.convert_tokens_to_ids(special_token)}")
    return tokenizer, model


def load_model(model_name: str, tokenizer_name: str, model_path: str, device, special_token='',lora_path=''):
    """
    load model and tokenizers from the given path
    model_path: can be one of the conditions below
        - if no fine-tuning, just the hf path either online or local
        - the fine-tuned model path if fine-tuned without lora
    lora_path: only add string when there is lora fine-tuning
    """
    if model_name in ['blenderbot','finetuned']:
        tokenizer = BlenderbotTokenizer.from_pretrained(tokenizer_name)
        # load model directly from path if not using lora
        if len(special_token) == 0:
            model = BlenderbotForConditionalGeneration.from_pretrained(model_path).to(device)
        # if the model is fine-tuned with lora; merge the additional lora layer
        elif len(lora_path) > 0:
            print('Applying the LORA fine-tuned model!!!')
            # reload the base LM for later generation
            model = BlenderbotForConditionalGeneration.from_pretrained(tokenizer_name).to(device)
            tokenizer, _ = add_special_token(special_token, tokenizer, model)
            model = PeftModel.from_pretrained(model, lora_path).to(device)
            model = model.merge_and_unload()

    elif model_name == 'qwen' or model_name == 'DialoGPT':
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device).eval()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # pad the tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model

def get_prompt1(
        role: str, month: int, include_len=True, dem=None,
        speaker_header=None,content_header=None,dem_turns=0, gen_length=50) -> str:
    '''
    Get the prompt based on different configurations with few-shot learning support.

    Args:
        role (str): Role of the speaker ('CHI' for child or 'ADULT' for parent)
        month (int): Age of the child in months
        include_len (bool): Whether to include length restriction in prompt
        dem (pd.DataFrame): Demonstration examples dataframe with columns ['speaker', 'utterance']
        dem_turns (int): Number of demonstration turns to include (n in n-shot learning)
        gen_length (int): Maximum length of generated response in words

    Returns:
        str: Formatted prompt including demonstrations if provided
    '''
    # Basic regulations for responses
    exp_reg = '<SILENCE> or <EMPTY> means that there is silence in previous turn. <NONSPEECH> or <UNINTELLIGIBLE> means that there is non-verbal or unintelligible speech. Only use correctly spelled words in your responses.'
    len_reg = f'Ensure your response is no longer than {str(gen_length)} words regardless of the prompt.'
    conv = 'Based on the given conversation history above'
    format = 'Do not output speaker label'
    # Construct few-shot demonstrations if provided
    demo_text = ""
    if dem is not None and dem_turns > 0:
        if len(dem) < dem_turns:
            raise ValueError(f"Requested {dem_turns} demonstration turns but only {len(dem)} examples available")

        # Get the first dem_turns examples
        demo_samples = dem.head(dem_turns)

        # Format demonstrations
        demo_text = "Here are some example conversations:\n\n"
        for idx, row in demo_samples.iterrows():
            demo_text += f"{row[speaker_header]}: {row[content_header]}\n"
        demo_text += "\nNow following these examples.\n"

    # Construct role-specific prompts
    base_prompt = (
        f"You are {'a ' + str(month) + '-month-old English-speaking child' if role == 'CHI' else 'the parent of a ' + str(month) + '-month-old English-speaking child'}. "
        f"Now you are having a conversation with your {'parent' if role == 'CHI' else 'child'}. "
        f"{exp_reg} "
        f"{len_reg if include_len else ''} "
        f"{demo_text}"  # Include demonstrations if available
        f"{conv}, give your response to {'parent' if role == 'CHI' else 'child'} input as {role}.{format}"
    )

    return base_prompt


def get_prompt(
        role: str,
        month: int,
        include_len=True,
        dem=None,
        speaker_header=None,
        content_header=None,
        dem_turns=0,
        gen_length=50,
        cot=False) -> str:
    exp_reg = '<SILENCE> or <EMPTY> means silence. <NONSPEECH> or <UNINTELLIGIBLE> means non-verbal/unintelligible speech. Use correctly spelled words.'
    len_reg = f'Response must be under {gen_length} words.'
    conv = 'Based on the conversation history above'
    format = 'Do not output speaker label'

    demo_text = ""
    if dem is not None and dem_turns > 0:
        if len(dem) < dem_turns:
            raise ValueError(f"Requested {dem_turns} demonstrations but only {len(dem)} examples available")

        demo_samples = dem.head(dem_turns)
        demo_text = "Examples:\n\n"
        for idx, row in demo_samples.iterrows():
            demo_text += f"{row[speaker_header]}: {row[content_header]}\n"
        demo_text += "\nFollow these examples.\n"

    role_desc = f"{'a ' + str(month) + '-month-old English-speaking child' if role == 'CHI' else 'the parent of a ' + str(month) + '-month-old English-speaking child'}"

    cot_prompt = "Think through this step by step:\n1. Consider the age-appropriate vocabulary and grammar\n2. Review the conversation context\n3. Formulate a natural response\n\n" if cot else ""

    base_prompt = (
        f"You are {role_desc}. "
        f"You are having a conversation with your {'parent' if role == 'CHI' else 'child'}. "
        f"{exp_reg} "
        f"{len_reg if include_len else ''} "
        f"{demo_text}"
        f"{cot_prompt}"
        f"{conv}, respond to {'parent' if role == 'CHI' else 'child'} input as {role}. {format}"
    )

    return base_prompt



def check_role(finetune_path:str,dialogue_role:str):
    print('Checking assigned roles!!!')
    ref_role = finetune_path.split('_')[-1]
    if ref_role == dialogue_role:
        print('Have assigned the correct role! ')
    else:
        raise Exception("Assigned the wrong role. Stopping execution.")
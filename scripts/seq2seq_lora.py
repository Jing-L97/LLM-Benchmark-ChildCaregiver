import torch
import pandas as pd
from datasets import Dataset
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from RL.setting import *
import os
import re
import sys
import argparse

def parseArgs(argv):
    # Run parameters
    parser = argparse.ArgumentParser(description='Finetune an encoder-decoder model')

    parser.add_argument('--model_name', type=str, default="facebook/blenderbot-400M-distill",
                        help='encoder-decoder model_name to be tuned')

    parser.add_argument('--train_path', type=str, default=f'{ROOT}/dataset/train/paired_added.csv',
                        help='Path to the train file')

    parser.add_argument('--gen_path', type=str, default=f'{ROOT}/dataset/eval/gen/seq2seq/blenderbot_single_seq2seq_lora.csv',
                        help='Path to the validation file')

    parser.add_argument('--finetuned_path', type=str, default=f'{ROOT}/model/single_turn_decoder_special',
                        help='Path to the model dir')

    parser.add_argument('--freeze_encoder', default=False,
                        help='Whether to freeze encoder during fine-tuning')

    parser.add_argument('--sent_num', default=1000,
                        help='the number of generated utterances')

    parser.add_argument('--eval_prop', default=0.05,
                        help='the prop of eval utterances')

    parser.add_argument('--special_token', default='<PAUSE>',
                        help='the prop of eval utterances')

    parser.add_argument('--max_length', default=10,
                        help='max number of tokens in one sent')

    parser.add_argument('--use_lora', default=True,
                        help='whether to use lora for high-efficient fine-tuning')

    parser.add_argument('--debug', default=True,
                        help='whether to debug with smaller dataset')

    return parser.parse_args(argv)


def load_dataset(filtered_frame:pd.DataFrame):
    """load dataset from the df"""
    data_dict = {}
    data_dict['input'] = filtered_frame['ADULT_utt'].tolist()
    data_dict['response'] = filtered_frame['CHI_utt'].tolist()
    dataset = Dataset.from_dict(data_dict)
    return dataset


def generate(tokenizer, model, input_text: str, max_length: int, device) -> str:
    # Tokenize the input and move it to the appropriate device
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    response_ids = model.generate(inputs['input_ids'], min_length=1, max_length=max_length)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response

def split_data(data:pd.DataFrame,eval_prop:float)->pd.DataFrame:
    """split the dataframe into train and eval"""
    # get the eval and train data respectively
    eval_df = data.iloc[:int(data.shape[0]*eval_prop)]
    train_df = data.iloc[int(data.shape[0]*eval_prop):]
    return train_df, eval_df


def add_special_token(special_token:str,tokenizer,model):

    # Add the special token to the tokenizer
    tokenizer.add_tokens([special_token])
    # Resize the model's token embeddings to accommodate the new token
    model.resize_token_embeddings(len(tokenizer))
    # Verify that the special token has been added
    print(f"Special token ID: {tokenizer.convert_tokens_to_ids(special_token)}")
    return tokenizer, model


def get_finetune_path(directory: str):
    # Initialize a variable to store the largest checkpoint
    largest_checkpoint = None
    largest_number = -1

    # Loop through each item in the directory
    for item in os.listdir(directory):
        # Check if the item matches the pattern 'checkpoint-<number>'
        match = re.match(r'checkpoint-(\d+)', item)
        if match:
            # Extract the number and convert it to an integer
            number = int(match.group(1))
            # Check if this is the largest number found so far
            if number > largest_number:
                largest_number = number
                largest_checkpoint = item

    # Output the largest checkpoint
    if largest_checkpoint:
        print(f"The largest checkpoint folder is: {largest_checkpoint}")
    else:
        print("No checkpoint folders found.")

    return largest_checkpoint





def main(argv):
    # Args parser
    args = parseArgs(argv)
    train_path = args.train_path
    gen_path = args.gen_path
    finetuned_path = args.finetuned_path
    eval_prop = args.eval_prop
    model_name = args.model_name
    sent_num = args.sent_num
    freeze_encoder = args.freeze_encoder
    special_token = args.special_token
    max_length = args.max_length
    use_lora = args.use_lora
    debug = args.debug
    # Check if a GPU is available, otherwise use the CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model and tokenizer
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    model = BlenderbotForConditionalGeneration.from_pretrained(model_name).to(device)

    if use_lora:
        print('Use LoRA to for high-efficient fine-tuning!')
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
        )
        print('Peft config set!')
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        print('Wrapping base LM with peft!')

    if len(special_token)>0:
       tokenizer, model= add_special_token(special_token,tokenizer,model)       
       print('Special token loaded!')

    if freeze_encoder:
        for param in model.get_encoder().parameters():
            param.requires_grad = False
        print('Freezing the encoder!')


    print('#########################################')
    print('The model and tokenizer have been loaded')
    print('#########################################')

    def tokenize(examples):
        # Tokenize the input and response
        inputs = tokenizer(examples['input'], max_length=128, truncation=True, padding="max_length")
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['response'], max_length=128, truncation=True, padding="max_length")
        inputs['labels'] = labels['input_ids']
        return inputs
    
    data_all = pd.read_csv(train_path)
    if debug:
        print('#########################################')
        print('Testing code with only 20 pairs of data!')
        print('#########################################')
        data_all = data_all.head(20)
        sent_num = 5

    gen_data = data_all.iloc[:sent_num]
    data = data_all.iloc[sent_num:]
    train_df, eval_df = split_data(data,eval_prop)

    # tokenize data
    train_dataset = load_dataset(train_df)
    tokenized_train = train_dataset.map(tokenize, batched=True)
    eval_dataset = load_dataset(eval_df)
    tokenized_eval = eval_dataset.map(tokenize, batched=True)


    print('#################################')
    print('The dataset has been tokenized')
    print('###############################')

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=finetuned_path,            # Directory to save results
        evaluation_strategy="epoch",       # Evaluate after each epoch
        learning_rate=5e-5,                # Learning rate
        per_device_train_batch_size=4,     # Batch size per device
        per_device_eval_batch_size=4,      # Batch size for evaluation
        num_train_epochs=3,                # Number of training epochs
        weight_decay=0.01,                 # Weight decay for optimization
        save_steps=500,                    # How often to save the model
        save_total_limit=2,                # Max number of model checkpoints to save
        logging_dir='wandb',              # Directory for logs
        logging_steps=10,                  # Log every 10 steps
    )

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,  # Use the tokenized training dataset
        eval_dataset=tokenized_eval    # For simplicity, use the same dataset for evaluation
    )

    # Start fine-tuning
    trainer.train()

    print('##########################')
    print('Finished finetuning!!!!')
    print('##########################')

    # Evaluate the model
    results = trainer.evaluate()
    print(results)
    print('##########################')
    print('Finished evaluating!!!!')
    print('##########################')

    # save the generations
    # reload ckp to make sure generating utt from the fine-tuned model
    # get the finetuned model path

    best_ckp = get_finetune_path(finetuned_path)
    if use_lora:
        # reload the base LM for later generation
        tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        model = BlenderbotForConditionalGeneration.from_pretrained(model_name).to(device)
        if len(special_token) > 0:
            tokenizer, model = add_special_token(special_token, tokenizer, model)
        print('base LM reloaded!')
        config = PeftConfig.from_pretrained(f'{finetuned_path}/{best_ckp}')
        finetuned_model = PeftModel.from_pretrained(model, f'{finetuned_path}/{best_ckp}')
        print('Loading fine-tuned peft model and base LM!')
    else:
        finetuned_model = BlenderbotForConditionalGeneration.from_pretrained(f'{finetuned_path}/{best_ckp}').to(device)

    print('Finetuned model loaded')

    gen_data['model_gen'] = gen_data['ADULT_utt'].apply(
        lambda text: generate(tokenizer, model, text, max_length, device))
    gen_data['finetuned_seq2seq'] = gen_data['ADULT_utt'].apply(
        lambda text: generate(tokenizer, finetuned_model, text, max_length, device))
    # create parent folder if the path does not exist
    parent_dir = os.path.dirname(gen_path)
    # Create the parent directory if it does not exist
    os.makedirs(parent_dir, exist_ok=True)
    gen_data.to_csv(gen_path)
    print('##########################')
    print('Finished generation!!!!')
    print('##########################')



if __name__ == "__main__":
    args = sys.argv[1:]
    main(args)

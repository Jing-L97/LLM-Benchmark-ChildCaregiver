
import re
import torch

def generate_beam(tokenizer,model,num_beams:int, max_length=20):
    #generate sequences with beam search
    input_text = " "
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    output = model.generate(input_ids,num_beams=num_beams ,do_sample=False,max_length=max_length)
    gen = tokenizer.decode(output[0], skip_special_tokens=True)
    return gen



def generate_topk(tokenizer,model,topk, max_length=20):
    #generate sequences with topk sampling
    input_text = " "
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    output = model.generate(input_ids,top_k =topk,max_length=max_length)
    gen = tokenizer.decode(output[0], skip_special_tokens=True)
    return gen

def generate_topp(tokenizer,model,topp, max_length=20):
    #generate sequences with topp sampling
    input_text = " "
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    output = model.generate(input_ids,top_p =topp,max_length=max_length)
    gen = tokenizer.decode(output[0], skip_special_tokens=True)
    return gen

def generate_sample(tokenizer,model,temp, max_length=30):
    #generate sequences with topp sampling
    input_text = " "
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    output = model.generate(input_ids,do_sample=True,temperature=temp,max_length=max_length)
    gen = tokenizer.decode(output[0], skip_special_tokens=True)
    return gen


def clean_text(text:str):
    """Replace punctuations (except apostrophes in contractions) with blanks"""
    # Define a regex pattern that matches all punctuation except apostrophes in contractions
    pattern = re.compile(r"[^\w\s']|(?<!\w)'|'(?!\w)")
    # Replace matched punctuations with blank
    processed_text = pattern.sub('', text)
    # Step 2: Replace newline characters with a space
    processed_text = processed_text.replace('\n', ' ')
    # Step 3: Split the string into words
    words = processed_text.split()
    # Step 4: Count the number of words
    word_count = len(words)
    return processed_text, word_count



def get_loss(tokenizer, model, gpu,device,sentence_good: str, sentence_bad: str) -> float:
        """get the loss of each pair"""

        # Encode sentences
        inputs_good = tokenizer(sentence_good, return_tensors="pt")
        inputs_bad = tokenizer(sentence_bad, return_tensors="pt")

        with torch.no_grad():
            # Calculate loss for good sentence
            outputs_good = model(**inputs_good, labels=inputs_good.input_ids)
            outputs_bad = model(**inputs_bad, labels=inputs_bad.input_ids)

            if gpu:
                outputs_good.to(device)
                outputs_bad.to(device)

            loss_good = outputs_good[0].item()
            loss_bad = outputs_bad[0].item()

        return loss_good, loss_bad


# get the boundary of rewards
def get_reward(df):
    """get the reward value in another column"""
    median_value = int(df['word_count'].median())
    # Create a new column based on the condition
    df['reward'] = df['word_count'].apply(lambda x: 1 if x > median_value else (-1 if x < median_value else 0))
    return df
    
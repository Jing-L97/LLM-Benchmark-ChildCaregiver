'''functions to evalute LMs'''
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
from scipy.stats import spearmanr
import spacy
import statsmodels.api as sm
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker
import numpy as np
from collections import Counter
import torch.nn.functional as F
from RL.setting import *
import re
# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

################################
# func to preperocess data
###############################

def is_word(word: str):
    """
    Check if a word is a legitimate English word.
    """
    spell = SpellChecker()
    return len(spell.unknown([word])) == 0


def remove_annotation(sentences):
    return [re.sub(r'\s*\(.*?\)\s*|\s*\*.*?\*\s*', ' ', sentence).strip() for sentence in sentences]


def revert_transcript(df_merged, header_lst, upper_header, lower_header,chosen_set):
    # Group by header_lst and get lists of sentences per speaker per group
    df_grouped = df_merged.groupby(header_lst + ['Speaker']).agg(list).reset_index()
    # Pivot to separate Speaker_A and Speaker_B into columns
    df_pivot = df_grouped.pivot(index=header_lst, columns='Speaker', values=chosen_set).reset_index()
    # Flatten the multi-index column from pivot
    df_pivot.columns.name = None

    # Get the lists of sentences for each speaker
    speaker_a_sentences = df_pivot[upper_header].apply(lambda x: x if isinstance(x, list) else [x])
    speaker_b_sentences = df_pivot[lower_header].apply(lambda x: x if isinstance(x, list) else [x])

    # Prepare a list to store the result
    new_rows = []

    # Iterate over the rows to generate multiple sentence pairs for each row
    for i in range(len(df_pivot)):
        a_sentences = speaker_a_sentences[i]
        b_sentences = speaker_b_sentences[i]
        max_len = max(len(a_sentences), len(b_sentences))

        # Ensure both lists have the same length by filling with empty strings if necessary
        a_sentences += [''] * (max_len - len(a_sentences))
        b_sentences += [''] * (max_len - len(b_sentences))

        # Create pairs and append to new_rows
        for a, b in zip(a_sentences, b_sentences):
            new_rows.append({
                **{col: df_pivot[col][i] for col in header_lst},  # Preserve other columns like TranscriptID, LineID
                upper_header: a,
                lower_header: b
            })

    # Create a DataFrame from the new rows
    df_sequential_pairs = pd.DataFrame(new_rows)

    return df_sequential_pairs


def reverse_prompt(data:pd.DataFrame,prompt_col:str,response_col:str):
    # reverse the prompt and response in one dialogue
    data[f'{prompt_col}_shifted'] = data[prompt_col].shift(-1)
    data = data.drop(prompt_col,axis=1)
    # get the rest of the header
    rest_header = list(data.columns)
    rest_header.remove(response_col)
    rest_header.remove(f'{prompt_col}_shifted')
    # shift other col by 1 index
    for col in rest_header:
        data[col] = data[col].shift(-1)
    # Select the relevant columns and drop the last row with NaN in A_shifted
    reorganized_data = data.dropna(subset=[f'{prompt_col}_shifted'])
    reorganized_data = reorganized_data.rename(columns={f'{prompt_col}_shifted': prompt_col})
    return reorganized_data


def convert_to_months(age_annotation):
    """convert CHILDES age annotation into months"""
    try:
        # Split the annotation into year and month parts
        year = age_annotation.split(';')[0]
        month = age_annotation.split(';')[1].split('.')[0]
        # Convert parts to integers
        year = int(year)
        month = int(month)
        # Convert to total months
        total_months = year * 12 + month
    except:
        total_months = 0
        print(age_annotation)
    return total_months


def load_transcript(data, header_lst, upper_header, lower_header):
    # List to store the new rows
    new_rows = []

    # Iterate row by row
    for index, row in data.iterrows():
        # Collect columns to duplicate
        duplicate_cols = {col: row[col] for col in header_lst}

        # Append row with upper_header (A) value
        new_rows.append({
            **duplicate_cols,
            'speaker': upper_header,
            'CHILDES': row[upper_header]
            
        })

        # Append row with lower_header (B) value
        new_rows.append({
            **duplicate_cols,
            'speaker': lower_header,
            'CHILDES': row[lower_header]
            
        })
    # Create a new DataFrame from the new_rows list
    df_merged = pd.DataFrame(new_rows)
    
    return df_merged




#########################
# func to get token stat
########################

WORD_PATTERN = re.compile(r'\b\w+\b')

class TokenCount:
    def __init__(self, data=None, name=None):
        if data is not None:
            self.df = pd.DataFrame(list(data.items()), columns=['word', 'count']).sort_values(by='count')
            self.name = name
        else:
            self.df = pd.DataFrame(columns=['word', 'count'])
        # check the existence of the columns below
        self.df['freq_m'] = self.df['count']/self.df['count'].sum() * 1000000
        self.df['correct']=self.df['word'].apply(is_word)

    def __str__(self):
        return self.df.to_string()

    def __repr__(self):
        return self.df.to_string()

    

    @staticmethod
    def from_df(file_path,header='word', name=None):
        """preprocess the input data"""
        if type(file_path)==str:
            lines = pd.read_csv(file_path)[header]
        elif isinstance(file_path, pd.DataFrame):   # in the case that the input is already a dataframe
            lines = file_path[header]
        # remove nan in the column
        lines = lines.dropna()
        # remove non-word descriptions in parenthesis
        lines = lines.astype(str)
        lines = remove_annotation(lines)
        word_counter = Counter()
        for line in lines:
            words = WORD_PATTERN.findall(line.lower())
            word_counter.update(words)
        return TokenCount(word_counter, header)


    def from_text_file_train(file_path):
        # Read from a text file and count words
        word_counter = Counter()
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # remove all blanks
                line = line.replace(" ", "")
                words = WORD_PATTERN.findall(line.lower())
                word_counter.update(words)
        basename = os.path.splitext(os.path.basename(file_path))[0]
        return TokenCount(word_counter, basename)

    @staticmethod
    def from_text_file(file_path):
        # Read from a text file and count words
        word_counter = Counter()

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                words = f.findall(line.lower())
                word_counter.update(words)
        basename = os.path.splitext(os.path.basename(file_path))[0]
        return TokenCount(word_counter, basename)
    

    def nonword(self):
        # Find the nonwords
        nonword_df = self.df[self.df['correct']==False]
        return TokenCount.from_df(nonword_df)


    def difference(self, othercorpus):
        # Find the words in df_ref that are not in df_gen using set difference
        missing_words = self.df.index.difference(othercorpus.df.index)
        # Extract the subset of df_ref with the missing words
        missing_words_df = self.df.loc[missing_words]
        # print("lengh df",len(self.df),"length other",len(othercorpus.df),"length difference",len(missing_words))
        return TokenCount.from_df(missing_words_df)

    def nb_of_types(self):
        # Return the number of unique words (types)
        return self.df.shape[0]

    def nb_of_tokens(self):
        # Return the sum of all word counts (nb of tokens=corpus size)
        return self.df['count'].sum()

    def zipf_coef(self):
        """Compute the zipf coefficient of a given token count."""
        sorted_data = np.sort(self.df["count"])
        nbpoints = sorted_data.shape[0]
        x = np.arange(1, (nbpoints + 1))  # ranks
        y = sorted_data[::-1]  # counts
        log_x = np.log(x)
        log_y = np.log(y)
        # Fit a linear regression model in the log-log space
        weights = 1 / x
        wls_model = sm.WLS(log_y, sm.add_constant(log_x), weights=weights)
        results = wls_model.fit()
        intercept = results.params[0]
        slope = results.params[1]
        log_y_fit = results.fittedvalues
        return log_x, log_y_fit, intercept, slope


    def stats(self):
        """Simple descriptive Statistics of the TokenCount (type/token, etc)"""

        if self.nb_of_tokens() != 0:
            typetok = self.nb_of_types() / self.nb_of_tokens()
        else:
            typetok = np.nan
        d = {'name': self.name, 'nb_token': self.nb_of_tokens(), 'nb_type': self.nb_of_types(), 'type/token': typetok}
        if self.nb_of_types() == 0:
            return d
        nb_hapaxes = np.sum(self.df['count'] == 1)
        nb_dipaxes = np.sum(self.df['count'] == 2)
        nb_le10 = np.sum(self.df['count'] <= 10)
        nb_nonword_type = np.sum(self.df['correct'] == False)
        nb_nonwords = self.df[self.df['correct'] == False]['count'].sum()
        d1 = {'nb_hapaxes': nb_hapaxes, 'p_hapaxes': nb_hapaxes / self.nb_of_types()}
        d2 = {'nb_dipaxes': nb_dipaxes, 'p_dipaxes': nb_dipaxes / self.nb_of_types()}
        d3 = {'nb_le_10': nb_le10, 'p_le_10': nb_le10 / self.nb_of_types()}
        sorted_data = np.sort(self.df["count"])
        top_count = sorted_data[-1]
        top_ge10_count = np.sum(sorted_data[-11:-1])
        d4 = {'prop_topcount': top_count / self.nb_of_tokens(),
              'prop_top_ge10_count': top_ge10_count / self.nb_of_tokens()}
        d5 = {'zipf_c': self.zipf_coef()[3]}
        d6 = {'nb_nonword_type': nb_nonword_type, 'p_nonword_type': nb_nonword_type / self.nb_of_types()}
        d7 = {'nb_nonword_token': nb_nonwords, 'p_nonword_token': nb_nonwords / self.nb_of_tokens()}
        return {**d, **d1, **d2, **d3, **d4, **d5, **d6, **d7}



def get_ttr(base_gen,header='cleaned_gen'):
    """get ttr from generations """
    base_word = TokenCount.from_df(base_gen, header)
    base_ttr = base_word.df.shape[0]/base_word.df['count'].sum()
    base_ttr_word = base_word.df[base_word.df['correct']==True].shape[0]/base_word.df[base_word.df['correct']==True]['count'].sum()
    return base_word, base_ttr, base_ttr_word



##########################
# func to get sent stat
##########################

def get_len(text:str):
    try:
        length = len(text.split())
    except:
        length = 0
    return length





def get_seg(count:pd.DataFrame,n_words:int,model, tokenizer,cal_prob=True):
    """get word segmentability"""
    # get the first n rows of the data
    if count.shape[0] > n_words:
        count = count.head(n_words)
    # get the avg prob of each group
    if cal_prob:
        count['log_prob'] = count['word'].apply(lambda x: get_prob(model, tokenizer, x))

    # get the number of words and non-words
    words = count[count['correct']==True]
    nwords = count[count['correct']==False]
    true_count = words.shape[0]
    false_count = nwords.shape[0]
    word_prob = words['log_prob'].mean()
    nword_prob = nwords['log_prob'].mean()
    # return a stat frame
    stat_frame = pd.DataFrame([true_count,false_count,word_prob,nword_prob]).T
    stat_frame.columns = ['word_num','nword_num','word_prob','nword_prob']
    
    return count, stat_frame



def get_freq_table(base_word,base_gen,model,tokenizer,n_word=10000,get_seg_stat=True):
    """get the freq table of the true words"""
    base_word.df['freq'] = base_word.df['freq_m']/ 1000000
    base_word_sorted = base_word.df.sort_values(by='freq',ascending=False)
    true_word_sorted = base_word_sorted[base_word_sorted['correct']==True]
    
    # only remove the suffix
    out_path = f'{'/'.join(base_gen.split('/')[:-1])}/{base_gen.split('/')[-1][:-4]}_stat.csv'
    base_word_sorted.to_csv(out_path)

    out_path = f'{'/'.join(base_gen.split('/')[:-1])}/{base_gen.split('/')[-1][:-4]}_true_stat.csv'
    true_word_sorted.to_csv(out_path)

    stat_frame = pd.DataFrame()
    # get the prob of each word type
    if get_seg_stat:
        base_word_annotated, stat_frame = get_seg(base_word_sorted,n_word,model,tokenizer)
        out_path = f'{'/'.join(base_gen.split('/')[:-1])}/{base_gen.split('/')[-1][:-4]}_stat_prob.csv'
        base_word_annotated.to_csv(out_path)
    
    return base_word_sorted,true_word_sorted, stat_frame



def get_ppl(test_path:str,tokenizer,model):

    """get ppl from test file"""
    with open(test_path,'r') as f:
        eval_sentences = f.readlines()
    
    # Ensure the model is in evaluation mode
    model.eval()
    # Initialize variables to calculate total loss and number of tokens
    total_loss = 0.0
    total_tokens = 0

    # Iterate over the evaluation sentences
    for sentence in eval_sentences:
        # Tokenize the input text
        inputs = tokenizer(sentence, return_tensors='pt', padding=True,max_length=2048)
        input_ids = inputs['input_ids']

        # Disable gradient calculation (inference mode)
        with torch.no_grad():
            # Get the model output including the loss
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

            # Accumulate total loss and count of tokens
            total_loss += loss.item() * input_ids.size(1)  # Multiply loss by the number of tokens
            total_tokens += input_ids.size(1)

    # Calculate the average loss across all tokens
    average_loss = total_loss / total_tokens
    # Calculate perplexity
    perplexity = torch.exp(torch.tensor(average_loss))
    return perplexity.item()



#### token count plots
def tc_plot(tokcount,title:str,set_type:str,color_dict:dict):
    """Zipf law plot"""
    # cumulative freq plot
    sorted_data = np.sort(tokcount.df["count"])
    nbpoints = sorted_data.shape[0]
    # Zipf law plot
    x = np.arange(1, (nbpoints + 1))  # ranks
    y = sorted_data[::-1]  # counts
    log_x, log_y_fit, intercept, slope = tokcount.zipf_coef()

    plt.plot(x, y, marker='o',color = color_dict[set_type],
             label=f'{set_type}: y = {slope:.2f}x + {intercept:.2f}')
    plt.plot(np.exp(log_x), np.exp(log_y_fit), color='grey')  # Regression line
    #plt.title(f'Zipf plot of LMs', fontweight='bold',fontsize=15)
    plt.xlabel('Rank', fontweight='bold',fontsize=15)
    plt.ylabel('Count', fontweight='bold',fontsize=15)
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, linestyle=':', linewidth=1, color='#bbbbbb') 
    # Add x-axis labels
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # Add legend
    plt.legend()


def plot_zipf(type_stat,model_type,set_type,color_dict):
    type_count = TokenCount()
    type_count.df = type_stat[type_stat['correct']==True]
    tc_plot(type_count,model_type,set_type,color_dict)



# load data

def get_stat(base_gen,header:str,set_type:str,model_type:str,test_path:str,tokenizer,model,n_word:int,color_dict,report_ppl=False,get_seg_stat=True):
    """get basic stat
    Input
        base_gen: path of the generated frame

    Returned columns
        utt_rate: prop of true sent
        word_type: prop of true word token type
        word_token: prop of true word tokens
        base_ttr: type/token
        base_ttr_word: type/token among true words
    """
    # add mean sent len
    gen_frame = pd.read_csv(base_gen)
    gen_frame['len'] = gen_frame[header].apply(get_len)
    gen_frame['sent_complexity'] = gen_frame[header].apply(get_syntax_complexity)
    sent_len = gen_frame['len'].mean()
    sent_complexity = gen_frame['sent_complexity'].mean()

    
    base_word, base_ttr, base_ttr_word = get_ttr(base_gen,header) 
    # get the stat
    base_stat,true_stat,seg_stat = get_freq_table(base_word,base_gen,model,tokenizer,n_word,get_seg_stat)
    word_type = true_stat.shape[0]/base_stat.shape[0]
    word_token = true_stat['count'].sum() / base_stat['count'].sum()

    # print out the top 10 words 
    print(set_type)
    print(true_stat[['word','count','freq']].head(10))
    print('#####################################')

    if report_ppl:
        # get ppl
        ppl = get_ppl(test_path,tokenizer,model)
        # output the stat 
        stat_frame = pd.DataFrame([set_type,ppl,sent_len,sent_complexity,word_type,word_token,base_ttr,base_ttr_word]).T
        stat_frame.columns = ['set_type','ppl','sent_len','sent_complexity','word_type','word_token','base_ttr','base_ttr_word']
    else:
        stat_frame = pd.DataFrame([set_type,sent_len,sent_complexity,word_type,word_token,base_ttr,base_ttr_word]).T
        stat_frame.columns = ['set_type','sent_len','sent_complexity','word_type','word_token','base_ttr','base_ttr_word']
    
    # only concatenate the seg stat when it is specified
    if get_seg_stat:  
        stat_frame = pd.concat([stat_frame, seg_stat], axis=1)
    
    # plot the zipf 
    plot_zipf(base_stat,model_type,set_type,color_dict)
    return stat_frame


##########################
# func for word counter
##########################
    
def get_prob(model, tokenizer, sentence:str):
    """get the prob of the given string """

    # Ensure model is in evaluation mode and does not track gradients
    model.eval()
    # Tokenize the sentence
    inputs = tokenizer(sentence, return_tensors='pt')

    # Get model outputs (logits)
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the logits for each token in the input sequence
    logits = outputs.logits

    # Calculate probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Get the predicted log probabilities for the actual tokens in the input
    input_ids = inputs["input_ids"]
    sequence_log_probs = log_probs[0, torch.arange(len(input_ids[0])), input_ids[0]]

    # Sum the log probabilities (for joint log-prob of the sequence)
    total_log_prob = torch.sum(sequence_log_probs)
    return total_log_prob.item()





##########################
# func to get word simi
##########################

def get_embedding(model, sentence: str, layer_num: int, tokenizer):
    # Tokenize input sentence
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Extract hidden states (all layers)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    num_layers = len(outputs.hidden_states)
    # Get the hidden states from the given layer
    if layer_num >= num_layers or layer_num < 0:
        raise IndexError(
            f"Layer number {layer_num} is out of range. Please choose a number between 0 and {num_layers - 1}.")

    hidden_states = outputs.hidden_states[layer_num]  # Shape: (batch_size, sequence_length, hidden_size)
    # Average the embeddings over the token dimension to get the word embedding
    embedding = hidden_states.mean(dim=1)  # Shape: (batch_size, hidden_size)
    return embedding


def get_distance(model, sentence1: str, sentence2: str, layer_num: int, tokenizer, normalize=True) -> float:
    # Get embeddings for both sentences
    embedding1 = get_embedding(model, sentence1, layer_num, tokenizer=tokenizer)
    embedding2 = get_embedding(model, sentence2, layer_num, tokenizer=tokenizer)

    # normalize the word embeddings
    if normalize:
        embedding1 = F.normalize(embedding1, p=2, dim=-1)
        embedding2 = F.normalize(embedding2, p=2, dim=-1)
    # Compute cosine similarity between the two sentence embeddings
    cos_sim = cosine_similarity(embedding1, embedding2)
    return cos_sim.item()



def load_sim_test(sim_root, file_name: str):
    if file_name == 'SimLex-999.txt':
        frame = pd.read_csv(sim_root + file_name, delimiter='\t')
        # rename column header
        frame.rename(columns={'SimLex999': 'human'}, inplace=True)

    elif file_name == 'SimVerb-3500.txt':
        frame = pd.read_csv(sim_root + file_name, delimiter='\t', header=None)
        frame.columns = ['word1', 'word2', 'POS', 'human', 'relation']

    return frame




def get_eval(sim_lex, header: str, layer_num: int, model, tokenizer) -> pd.DataFrame:
    cor_lst = []
    p_lst = []
    # generate the layer_num_lst
    layer_num_lst = list(range(1, layer_num + 1))
    for layer_num in tqdm(layer_num_lst):
        # calculate the distance
        sim_lex[f"{header}_" + str(layer_num)] = sim_lex.apply(
            lambda row: get_distance(model, row['word1'], row['word2'], layer_num, tokenizer=tokenizer), axis=1)
        # get simi
        correlation, p_value = spearmanr(sim_lex['human'], sim_lex[f"{header}_" + str(layer_num)])
        # get the max num from the stat
        cor_lst.append(correlation)
        p_lst.append(p_value)
    base_stat = pd.DataFrame([layer_num_lst, cor_lst, p_lst]).T
    base_stat.columns = ['layer_num', 'correlation', 'p_value']
    return sim_lex, base_stat


def check_value(value, lst):
    return 1 if value in lst else 0


def select_eval(sem_eval,train_path):

    '''divide eval set into ind, ood and rest'''
    # get the train word lst
    word_counter = TokenCount.from_text_file(train_path)
    word_lst = word_counter.df['word'].tolist()

    sem_eval['word1_presence'] = sem_eval['word1'].apply(check_value, lst=word_lst)
    sem_eval['word2_presence'] = sem_eval['word2'].apply(check_value, lst=word_lst)
    # select the words appearing in the training set
    ind_set = sem_eval[(sem_eval['word1_presence'] == 1) & (sem_eval['word2_presence'] == 1)]
    ood_set = sem_eval[(sem_eval['word1_presence'] == 0) & (sem_eval['word2_presence'] == 0)]
    # get the rest of the dataframe
    rest_set = sem_eval[~sem_eval.index.isin(ind_set.index) & ~sem_eval.index.isin(ood_set.index)]

    return ind_set, ood_set, rest_set




def get_mean_stat(df:pd.DataFrame,group_col:str, leftmost_col:str):
    """
    get mean stat by a group column
    arguments: the stat df; 
    the column to group the stat: if it's empty string, do not cluster by this group 
    the left most column
    return dfs of means and std
    """
    columns_right = df.columns[df.columns.get_loc(leftmost_col):]
    # Step 2: Group by the 'Group' column and get the mean of the columns to the right
    if len(group_col) > 0: 
        df = df.groupby(group_col)[columns_right].mean()
    # Step 3: Get the total mean across groups for each column
    total_means = df.mean()
    # Step 4: Get the standard deviation across group means for each column
    group_stds = df.std()
    return pd.DataFrame([total_means]), pd.DataFrame([group_stds])
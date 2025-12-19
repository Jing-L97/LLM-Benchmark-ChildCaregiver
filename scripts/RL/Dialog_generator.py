import ollama
import copy
import random
from RL.setting import *
from RL.gen_util import *
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import openai
client = OpenAI()
'''
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",  # "http://<Your api-server IP>:port"
    api_key="sk-no-key-required",
)
'''
##########################
# model para settings
##########################

nonLLM_lst = ['blenderbot','DialoGPT','finetuned']
LLM_lst = ['llama3','mistral-nemo','mistral','qwen2.5','phi3.5']
gpt_lst = ["chatgpt-4o-latest",'gpt-3.5-turbo-0125']

blender_para = {'model_name':'blenderbot',
                'tokenizer_name':"facebook/blenderbot-400M-distill",
                'model_path':"facebook/blenderbot-400M-distill"}

finetuned_para = {'model_name':'finetuned',
                'tokenizer_name':"facebook/blenderbot-400M-distill"}

DialoGPT_para = {'model_name':'DialoGPT',
                'tokenizer_name':"microsoft/DialoGPT-medium",
                'model_path':"microsoft/DialoGPT-medium"}

mistral_para = {'model_name': 'mistral'}       # mistral 7B
mistral_nemo_para = {'model_name': 'mistral-nemo'}      # mistral 12B
qwen_para = {'model_name': 'qwen2.5'}      # qwen 7.6B
llama3_para = {'model_name': 'llama3'}         # llama3 8B
gpt4_para = {'model_name': "chatgpt-4o-latest"}
gpt3_para = {'model_name': 'gpt-3.5-turbo-0125'}
nemotron_para = {'model_name': 'nemotron-mini'}
mixtral_para = {'model_name': 'mixtral'}
phi3_5_para = {'model_name':'phi3.5'}

model_para_lst = [qwen_para,blender_para,DialoGPT_para,mistral_para,llama3_para,nemotron_para,mixtral_para,phi3_5_para,
                  mistral_nemo_para,finetuned_para,gpt4_para,gpt3_para]

role_dict = {'CHI':'ADULT', 'ADULT':'CHI'}
len_dict = {'CHI':6, 'ADULT':50}
sil_dict = {'CHI':0.315, 'ADULT':0.037}

##########################
# model para settings
##########################
class TextGenerator:
    def __init__(self, model_name: str, tokenizer, model, device='cpu'):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def generate_ref(self, input_text: str, max_length=10) -> str:
        # Tokenize the input and move it to the appropriate device
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(self.device)
        response_ids = self.model.generate(inputs['input_ids'], min_length=1, max_new_tokens=max_length)
        response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)
        return response

    def generate_qwen(self, prompt: str, system_role: str, max_new_tokens=12) -> str:
        messages = [
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def generate_LLM(self, messages: list) -> str:
        """
        Generate the results from the conversational history
        input:
          - conversation history
        return:
          - generated results
          - conversation history
        """
        response = ollama.chat(self.model_name, messages=messages)
        message = response['message']
        messages.append(message)
        return message['content'], messages


    def generate_gpt(self, messages: list, max_tokens=None) -> str:
        """Generate based on text prompt"""
        if max_tokens==None:
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )

        else:
            completion = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens
        )
        return completion.choices[0].message.content

    def generate(self, initial_input: str, messages: list, max_token=12, sil_prob=0) -> str:
        """
        Generate the input from any type of input with a probability `n` that generation is <SILENCE>.
        Format the messages into required formats, e.g. {'role': 'system', 'content': system_role}.
        """
        # Determine if we generate <SILENCE> based on the probability `sil_prob`
        if random.random() < sil_prob:
            return '<SILENCE>'
        else:
            if self.model_name in ['blenderbot', 'DialoGPT', 'finetuned']:
                    gen = self.generate_ref(initial_input, max_length=max_token)
            elif self.model_name == 'qwen':
                    gen = self.generate_qwen(initial_input, 'system_role', max_new_tokens=max_token)
            else:
                # Add the system_role related prompt at the beginning
                if self.model_name in LLM_lst:
                    gen, messages = self.generate_LLM(messages)
                elif self.model_name == "chatgpt-4o-latest":
                    gen = self.generate_gpt(messages)
            return gen


##########################
# model para settings
##########################

class DialogueSimulator:
    def __init__(self, child_model_name, adult_model_name, child_tokenizer, child_model, adult_tokenizer, adult_model,
                 sil_dict, device, child_prompt, adult_prompt, max_child_token=10, max_adult_token=12,
                 child_history_turns=1, adult_history_turns='all', num_turns='all'):
        # Initialize the DialogueSimulator with required parameters
        self.child_model_name = child_model_name
        self.adult_model_name = adult_model_name
        self.child_tokenizer = child_tokenizer
        self.child_model = child_model
        self.adult_tokenizer = adult_tokenizer
        self.adult_model = adult_model
        self.sil_dict = sil_dict
        self.device = device
        self.child_prompt = child_prompt
        self.adult_prompt = adult_prompt
        self.max_child_token = max_child_token
        self.max_adult_token = max_adult_token
        self.child_history_turns = child_history_turns
        self.adult_history_turns = adult_history_turns
        self.num_turns = num_turns

        # Create TextGenerator instance for both child and adult models
        self.child_generator = TextGenerator(child_model_name, child_tokenizer, child_model, device)
        self.adult_generator = TextGenerator(adult_model_name, adult_tokenizer, adult_model, device)

    def format_conversation_history(self, history: list, model_name: str, max_turns=4):
        '''
        Select previous turns into conversation history
        '''
        if model_name in LLM_lst:
            # Insert the system role; revise the previous generation speaker label
            if max_turns == 'all' or len(history) < max_turns-1:
                messages = history
            elif len(history) >= max_turns-1:
                messages = history[-(max_turns-1):]  # Limit the conversation history
            return messages

        elif model_name in gpt_lst:
            # Insert the system role; revise the previous generation speaker label
            if max_turns == 'all' or len(history) < max_turns:
                messages = history
            elif len(history) >= max_turns:
                messages = history[-(max_turns):]  # Limit the conversation history
            return messages

        else:
            if max_turns or len(history) < max_turns:  # concatenate all conversation history
                pass
            elif len(history) >= max_turns:
                history = history[-max_turns:]  # Limit the conversation history
            else:
                print('Not enough history turns')
            return " ".join(history)

    def generate_response(self, model_name: str, prior_turn: str, role_prompt: str, conversation_log: list,
                          message_dict: dict, max_token: int, role: str, history_num: int):
        '''
        Generate response to prior turn and based on the conversation history
        '''
        try:
            if model_name in gpt_lst:
                messages = message_dict['gpt']


            elif model_name in LLM_lst:
                messages = message_dict['ollama']

            message_input = copy.deepcopy(messages)
            message_input = self.format_conversation_history(message_input, model_name, history_num)


            if model_name in LLM_lst:
                message_input.append({'role': 'user', 'content': f'{role_prompt}: {prior_turn}'})

            elif model_name in gpt_lst:
                message_input.append({'role': 'system', 'content': role_prompt})
            print(message_input)
        except:
            message_input = []
            print(f'Skip formatting input message as we use {model_name} model')

        input_text = self.format_conversation_history(conversation_log, model_name, history_num)

        # Now use TextGenerator's generate method to get the response
        response = self.child_generator.generate(input_text, message_input, max_token, sil_prob=self.sil_dict[role]) \
            if role == 'CHI' else \
            self.adult_generator.generate(input_text, message_input, max_token, sil_prob=self.sil_dict[role])

        print(f'{role}: {response}')

        conversation_log.append(f"{response}")
        message_dict['ollama'].append({'role': role_dict[role], 'content': prior_turn})

        role_label = 'system' if role == 'ADULT' else 'user'
        message_dict['gpt'].append({'role': role_label, 'content': response})

        return conversation_log, message_dict, response

    def simulate_conversation(self, child_response: str) -> list:
        '''
        Build single dialogue based on initial prompt
        '''
        conversation_log = [child_response]
        message_dict = {'ollama': [], 'gpt': [{'role': 'user', 'content': child_response}]}
        print('############################################')
        print(f'CHI: {child_response}')

        n = 0
        for turn in range(self.num_turns):
            # Generate adult's response using TextGenerator
            conversation_log, messages, adult_response = self.generate_response(
                self.adult_model_name, child_response, self.adult_prompt, conversation_log, message_dict,
                self.max_adult_token, 'ADULT',self.adult_history_turns)

            # Generate child's response using TextGenerator
            conversation_log, messages, child_response = self.generate_response(
                self.child_model_name, adult_response, self.child_prompt, conversation_log, message_dict,
                self.max_child_token, 'CHI',self.child_history_turns)

            if len(adult_response.split()) > 50:
                print(f'############### {len(adult_response.split())} words')
                n += 1

        print(f'EXCEEDING TURNS: {str(n)}')
        return conversation_log

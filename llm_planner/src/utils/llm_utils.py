import json
from typing import List, Dict
from openai import OpenAI
from .get_dict import *

class Planner():
    def __init__(self, openai_api_key: str, system_prompt_path: str, map_path: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.system_prompt_path = system_prompt_path
        self.map_path = map_path

    def _parser(self, message: str, K: int = 3) -> List:
        response = message.split('Command:')
        commands = response[-1].strip().split('\n')
        self.command_history = self.command_history  + commands 
        self.command_history = self.command_history[-3:]
        return commands
    
    def create_plan(self, data: Dict, instruction: str) -> List:
        self.instruction = instruction
        self.system_prompt = load_txt_file(self.system_prompt_path)
        transformed_dict = transform_dictionary(data)
        transformed_dict = convert_sets_to_lists(transformed_dict)
        dict_string = json.dumps(transformed_dict, indent=4)
        print(dict_string)
        current_user_prompt =  f'''
Map:    
{dict_string}
Command History:
None
Instruction:
{instruction}'''
        prompt = [
        {
        "role": "system",
        "content": self.system_prompt
        },
        {
        "role": "user",
        "content": current_user_prompt
        }
        ]
        response = self.client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=prompt,
                        temperature=0,
                        max_tokens=256,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                        ).choices[0].message.content
        self.command_history = []
        return self._parser(response)
    
    def update_plan(self, data: Dict, feedback: str):
        # data = load_yaml_file(self.map_path)
        transformed_dict = transform_dictionary(data)
        transformed_dict = convert_sets_to_lists(transformed_dict)
        dict_string = json.dumps(transformed_dict, indent=4)
        print(dict_string)
        current_user_prompt = f'''
Map:    
{dict_string}
Command History:
{self.command_history}
Feedback:
{feedback}
Instruction:
{self.instruction}'''
        prompt = [
        {
        "role": "system",
        "content": self.system_prompt
        },
        {
        "role": "user",
        "content": current_user_prompt
        }
        ]
        response = self.client.chat.completions.create(
                        model="gpt-4-turbo",
                        messages=prompt,
                        temperature=0,
                        max_tokens=256,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                        ).choices[0].message.content
        return self._parser(response)

def check_instruction(instruction: str):
    out = input(f'Is the instruction "{instruction}" completed? y/n')
    if out.strip().lower() in ['yes', 'y']:
        return True
    else:
        return False
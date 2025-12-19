from typing import Dict, Tuple
import argparse
import re
import json
import re
import paramiko
import os
from api import GoalPub
import math

from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('jetson-ip', type=str, help='IP address of the external Jetson', required=True)
    parser.add_argument('jetson-username', type=str, help='username for the external Jetson', required=True)
    parser.add_argument('jetson-password', type=str, help= 'Password for the external Jetson', required=True)
    parser.add_argument('map-path', type=str, help='Json map path in the external Jetson', required=True)
    parser.add_argument('openai-key', type=str, help='Openai key for the LLM Planner', required=True)
    return parser.parse_args()

def wait_instruction() -> str:
    instruction = input('Your instruction: ')
    return instruction


def send_commmand(command: str, scene_map: Dict, instruction: str, api: GoalPub) -> Tuple[bool,bool]:
    simplified_map = simplify_structure(scene_map)
    command = command.strip()
    if command.startswith('goto'):
        matches = re.findall(r'"(.*?)"', command)
        if len(matches) == 0:
            matches = re.findall(r"'(.*?)'", command)
        if len(matches) == 0:
            inside_paranthesis = command[command.index('(')+1:command.index(')')]
            matches = [value.strip() for value in inside_paranthesis.split(',')]
        if len(matches) == 3:
            x, y = simplified_map['buildings'][matches[0]]['floors'][matches[1]]['rooms'][matches[2]]['location']
        else:
            x, y = simplified_map['buildings'][matches[0]]['floors'][matches[1]]['rooms'][matches[2]]['objects'][matches[3]]['location']
        success = api.execute_goto(x, y)
        is_completed = check_instruction(instruction)
    elif command.startswith('search_room'):
        matches = re.findall(r'"(.*?)"', command)
        if len(matches) == 0:
            matches = re.findall(r"'(.*?)'", command)
        if len(matches) == 0:
            inside_paranthesis = command[command.index('(')+1:command.index(')')]
            matches = [value.strip() for value in inside_paranthesis.split(',')]
        if len(matches) == 3:
            x, y = simplified_map['buildings'][matches[0]]['floors'][matches[1]]['rooms'][matches[2]]['location']
        else:
            x, y = simplified_map['buildings'][matches[0]]['floors'][matches[1]]['rooms'][matches[2]]['objects'][matches[3]]['location']
        success = api.execute_goto(x, y)
        success = api.execute_explore_locally()
        is_completed = False
    elif command.startswith('explore_globally()'):
        success = api.execute_explore_globally(60)
        is_completed = False
    else:
        raise Exception(f'{command} is not defined in the controller.')
    return success, is_completed


def get_latest_json(remote_directory, args):
    # Create an SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    os.makedirs('./tmp', exist_ok = True)
    # Initialize the SFTP client
    sftp = None
    hostname = args.jetson_ip  
    port = 22
    username = args.jetson_username
    password = args.jetson_password
    try:
        # Connect to the remote host
        ssh.connect(hostname, port, username, password)
        sftp = ssh.open_sftp()
        print("Connected to remote system")
        files = sftp.listdir(remote_directory)
        json_files = [f for f in files if re.match(r'^\d+.json$', f)]

        if not json_files:
            print("No JSON files found in the directory.")
            return None

        latest_file = sorted(json_files, key=lambda x: int(x.split('.')[0]))[-1]
        remote_path = f"{remote_directory}/{latest_file}"
        local_path = os.path.join('./tmp',latest_file)
        sftp.get(remote_path, local_path)
        with open(local_path, 'r') as file:
            json_data = json.load(file)

        print("Latest JSON file downloaded and loaded successfully.")
        return json_data

    except Exception as e:
        print(f"Failed to download and load JSON file: {e}")
        return None

    finally:
        if sftp:
            sftp.close()
        ssh.close()

def get_scene_map(map_path):
    scene_map = get_latest_json(map_path)
    return scene_map


if __name__ == '__main__':
    args = get_args()    
    system_prompt_path = './prompts/system_prompt.txt'
    user_prompt_path = './prompts/user_prompt.txt'
    
    map_path = args.maps_path

    api_key = args.openai_key
    api = GoalPub()
    # Execute explore locally in the beginning of the code.
    api.execute_explore_locally(theta=math.pi/2, rotations=4)
    planner = Planner(openai_api_key=api_key,
                      system_prompt_path=system_prompt_path,
                      user_prompt_path=user_prompt_path,
                       map_path=map_path)
    while 1: #outer loop
        instruction = wait_instruction()
        scene_map = get_scene_map(map_path)
        commands = planner.create_plan(scene_map, instruction)
        print('The given commands by the LLM are', commands)
        # ask user if the commands are fine to run, if the input is no, stop the code
        user_input = input("Are the commands fine to run? (yes/no): ")
        if user_input.lower() == 'no':
            print("Stopping the program.")
            exit()
        else:
            print("Continuing with the program.")
        is_completed = False
        while not is_completed: # Inner loop
            print('hi1')
            for command in commands:
                print('hi_command')
                #Before executing each command, get the updated map.
                scene_map = get_scene_map(map_path)
                try:
                    success, is_completed = send_commmand(command, scene_map, instruction, api)
                except Exception as e:
                    success = False
                    is_completed = False
                    feedback = e
                    print(f'Command {command} cannot be executed. It will be given as a feedback to the LLM\nFeedback is {feedback}')
                    break
                if  (not success) and (not is_completed):
                    feedback = f'{command} could not be executed by the robot.'
                    break
                if is_completed:
                   break
            if success and (not is_completed):
                #All the commands executed without an error but objective's been completed yet.
                feedback = 'Instruction has not been satisfied yet.'


            if not is_completed:
                #Ask a new set of commands.
                scene_map = get_scene_map(map_path)
                commands = planner.update_plan(scene_map,feedback)
                print('The given commands by the LLM are', commands)
                user_input = input("Are the commands fine to run? (yes/no): ")
                if user_input.lower() == 'no':
                    print("Stopping the program.")
                    exit()
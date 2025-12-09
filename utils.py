import csv
import re
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import numpy as np
from itertools import product
import xlsxwriter


def env_name_resolution(folders_name):

    current_folder = re.search(r'(?<=Env_).*', str(folders_name)).group(0)

    name = re.split('_', current_folder) 

    # Concat env property names
    new_list = []
    counter = 0
    for ind_elem, elem in enumerate(name):
        if elem[0].isalpha():
            if counter >= 1:
                counter -= 1
                continue
            else:
                word_with_upper_case = elem[0].upper() + elem[1:]
                counter = 1
                while(name[ind_elem+counter].isalpha()):
                    word_with_upper_case += ' ' + name[ind_elem+counter][0].upper() + name[ind_elem+counter][1:]
                    counter +=1
                new_list.append(word_with_upper_case)
        else:
            new_list.append(elem)
            counter = 0

    final_list = []
    for ind_elem, elem in enumerate(new_list):
        if elem[0].isalpha():
            if elem == 'Grid Shape':
                prop_name = elem + ' ' + str(new_list[ind_elem+1]) + 'x' + str(new_list[ind_elem+2])

            elif elem == 'Max Steps In Episode':
                prop_name = elem + ' ' + str(new_list[ind_elem+1])

            # All float values
            else:
                prop_name = elem + ' ' + str(new_list[ind_elem+1]) + '.' + str(new_list[ind_elem+2])

            final_list.append(prop_name)

    # Creation of final word
    env_name = 'Environment: '

    for ind_prop, prop in enumerate(final_list):
        if (len(final_list)-1) == ind_prop:
            env_name += prop
        else:
            env_name += prop + ', '

    return env_name


def get_data(hp_param, experiment_name):

    # Experiment folder
    folder = Path('results/' + experiment_name)
    env_folder = [env_folder for env_folder in folder.iterdir() if env_folder.is_dir() and env_folder.name.startswith("Env")]
    hp_folder = [hp_folder for hp_folder in env_folder[0].iterdir() if hp_folder.name == f'Hp_{hp_param}' and hp_folder.is_dir()]
    value_folder = [value_folder for value_folder in hp_folder[0].iterdir()]
    
    number_envs = len(env_folder)
    number_values = len(value_folder)
    number_seeds = sum(1 for f in value_folder[0].iterdir() if f.is_dir())

    data = np.zeros((number_envs, number_values, number_seeds))

    for ind_env, env_folder in enumerate(env_folder):

        ### Iterate over environment folders:
        for hp_folder in env_folder.iterdir():

            if hp_folder.name == f'Hp_{hp_param}':
                ### Case: hp folder equal

                for ind_value, value_folder in enumerate(hp_folder.iterdir()):

                ### in dem Folder sind alle values: Iterate 

                    for ind_seed, seed_folder in enumerate(value_folder.iterdir()):

                        # Performance Value
                        performance = pd.read_csv(f'{seed_folder}/performance.csv').columns[0]

                        data[ind_env][ind_value][ind_seed] = performance

    return data

def get_data_no_hp(experiment_name):

    # Experiment folder
    folder = Path('results/' + experiment_name)

    env_folders = [env_folder for env_folder in folder.iterdir() if env_folder.is_dir() and env_folder.name.startswith("Env")]
    seed_folder = [seed_folder for seed_folder in env_folders[0].iterdir()]
    
    number_envs = len(env_folders)
    number_seeds = len(seed_folder)

    data = np.zeros((number_envs, number_seeds))

    for ind_env, env_folder in enumerate(env_folders):

        if env_folder.is_dir() and env_folder.name.startswith("Env"):
            ### Iterate over environment folders:

            for ind_seed, seed_folder in enumerate(env_folder.iterdir()):
            ### in dem Folder sind alle values: Iterate 
                performance = pd.read_csv(f'{seed_folder}/performance.csv').columns[0]

                data[ind_env][ind_seed] = performance

    return data

    
def change_format(hp):
    new_format = ' '.join(word.capitalize() for word in hp.split('_'))
    return new_format


def collecting_results(folder, mode='xlsx'):

    # 1. Step: Data FRame erzeugen und danach in csv speichern
    result_dir = Path('results/' + folder)
    exp_dir = Path('examples/configs/' + folder + '/info.yaml')
    
    cfg = OmegaConf.load(exp_dir)

    # Convert to Python dict
    info_dict = OmegaConf.to_container(cfg, resolve=True)

    # 
    hp_list = list(info_dict['hp'].keys())
    number_seeds = info_dict['seeds']

    env_names = []

    ### Mistake here:
    ## Use name of the folder --> folder saved in different order than given in grid search file

    # Do not iterate over grid_search_items instead try to read name of folders
    ## Read in properties --> from filenames
    env_folders = [env_folder for env_folder in result_dir.iterdir() if env_folder.is_dir() and env_folder.name.startswith("Env")]
    env_names = [env_name_resolution(env_folder.name) for env_folder in env_folders]

    # Create excel file here: und  dann für jeden HP ein eigenes sheet
    writer = xlsxwriter.Workbook(f'{result_dir}/performances.xlsx')

    all_sheets = []
    ## Case Distinction: No hyperparamter:
    if len(hp_list):
        all_sheets = []
        for ind_hp, hp in enumerate(hp_list):
            first_row = []
            data = get_data(hp, result_dir)
            first_row.append(f'Environment / Hyperparameter: {change_format(hp)}')

            for val in info_dict['hp'][hp]:
                first_row.append(f'Value: {val}')
            all_rows = [first_row]

            for ind_env, env in enumerate(env_names):
                new_row = [env]
                ### Fails when only one seed is used ##
                if number_seeds > 1:
                    data_row = [', '.join(map(str, [round(val,2) for val in values])) for values in data[ind_env]]
                else:
                    data_row = [val for val in data[ind_env]][0]
                new_row.extend(data_row)

                all_rows.append(new_row)
        
            df = pd.DataFrame(all_rows)
            all_sheets.append(df)
    else:
        first_row = ['Environment / Seeds']
        first_row.extend([f'Seed {i}' for i in range(number_seeds)])
        all_rows = [first_row]

        data = get_data_no_hp(result_dir)

        for ind_env, env in enumerate(env_names):
            new_row = [env]
            data_row = [round(val,2) for val in data[ind_env]]
            new_row.extend(data_row)

            all_rows.append(new_row)
    
        df = pd.DataFrame(all_rows)
        all_sheets.append(df)

    if mode == 'csv':
        pass
        ### TODO

    elif mode == 'xlsx':

        with pd.ExcelWriter(f'{result_dir}/performances.xlsx', engine="xlsxwriter") as writer:
            if len(hp_list):
                for ind, df in enumerate(all_sheets):
                    df.to_excel(writer, sheet_name=f'{change_format(hp_list[ind])}', index=False)
            else:
                all_sheets[0].to_excel(writer, sheet_name='Default_HP', index=False)
    
    else:
        raise ValueError('Invalid mode')

if __name__ == "__main__":
    collecting_results("Test_again")

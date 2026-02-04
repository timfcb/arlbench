import csv
import re
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import numpy as np
from itertools import product
import xlsxwriter


def results_to_csv(hp_values, different_seeds, result_folder, all_config_ids): 
    ### Save values in CSV for each hyperparameter
    for hp_index, hp in enumerate(hp_values.keys()):

        with open(f'{result_folder}/{hp}.csv', 'w') as f:

            # First Row Specifications
            first_row = 'ENV_ID,'
            hp_value_range = hp_values[hp]
            for ind_val, val in enumerate(hp_value_range):
                for seed in different_seeds:
                    first_row += f'Value_{val}_Seed_{seed},'

            first_row = first_row[:-1] + '\n'
            f.write(first_row)

            for config_id in all_config_ids:
                data = np.load(f'{result_folder}/config_{config_id}.npz')

                hp_data = data[f'arr_{hp_index}']

                row_entry = f'{config_id},'
                for val in hp_data.flatten():
                    row_entry += f'{val},'
                entry = row_entry[:-1] + '\n'
                f.write(entry)


# INFO: results is list of list of values
def save_results(result_path, results, different_seeds):

    with open(f'{result_path}/performances.csv', 'w') as f:

        # First Row Specifications
        first_row = 'EXP_ID,'
        for seed in different_seeds:
            first_row += f'Seed_{seed},'

        first_row = first_row[:-1] + '\n'
        f.write(first_row)

        # Append results to csv file
        for config_id, env in enumerate(results):

            for hp_id, elem in enumerate(env):

                row_entry = ''
                for val in elem:
                    rounded_val = round(val,2)
                    row_entry += f'{rounded_val},'
                entry = row_entry[:-1]
                row_label = f'ENV_ID_{config_id}_HP_ID_{hp_id}'
                f.write(f'{row_label},{entry}\n')


def mapping_id_to_exp(current_exp, number_seeds):

    config_number, seed = divmod(current_exp, number_seeds)

    return config_number, seed

# Returns info about experiment parameters which is created at the beginning
def load_info(experiment_name):

    exp_dir = Path('examples/configs/' + experiment_name + '/info.yaml')
    
    cfg = OmegaConf.load(exp_dir)

    # Convert to Python dict
    info_dict = OmegaConf.to_container(cfg, resolve=True)

    return info_dict

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

def get_data(experiment_name, hp_name, hp_values, number_seeds):

    path = f'results/{experiment_name}/{hp_name}.csv'
    df_data = pd.read_csv(path, sep=',')
    number_envs = df_data.shape[0]

    data = np.zeros((number_envs, len(hp_values), number_seeds))
    for ind_env in range(number_envs):

        for ind_value in range(len(hp_values)):

            for ind_seed in range(number_seeds):

                performance = df_data.iloc[ind_env, ind_value * number_seeds + ind_seed]
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

def remove_points(val):

    if isinstance(val, list):
        val_string = str(val)
        modified = val_string.replace('[', '').replace(']', '').replace(', ', '_')
    else:
        modified = val
        str_val = str(val)
        if '.' in str_val:
            modified = str_val.replace('.', '_')

    return modified

def collecting_results(folder, mode='xlsx'):

    # 1. Step: Data FRame erzeugen und danach in csv speichern
    result_dir = Path('results/' + folder)

    # Convert to Python dict
    info_dict = load_info(folder)

    # Get hp_list
    hp_list = list(info_dict['hp'].keys())
    number_seeds = info_dict['seeds']

    env_names = []

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
            data = get_data(hp, folder)
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

        data = get_data_no_hp(folder)

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

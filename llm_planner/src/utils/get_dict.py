import yaml


def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def load_txt_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read().strip()
    return data

def remove_location_coordinates(d):
    if isinstance(d, dict):
        return {k: remove_location_coordinates(v) for k, v in d.items() if k != 'location'}
    elif isinstance(d, list):
        return [remove_location_coordinates(v) for v in d]
    else:
        return d

def simplify_structure(d):
    if isinstance(d, list) and len(d) == 1 and isinstance(d[0], dict) and not d[0]:
        return {}
    elif isinstance(d, list):
        return [simplify_structure(item) for item in d]
    elif isinstance(d, dict):
        new_dict = {}
        for key, value in d.items():
            simplified_value = simplify_structure(value)
            if isinstance(simplified_value, list) and all(isinstance(item, dict) for item in simplified_value):
                merged_dict = {}
                for item in simplified_value:
                    merged_dict.update(item)
                new_dict[key] = merged_dict
            else:
                new_dict[key] = simplified_value
        return new_dict
    else:
        return d

def objects_to_set(d):
    if isinstance(d, dict):
        new_dict = {}
        for key, value in d.items():
            if key == 'objects':
                if value is None:
                    new_dict[key] = set()
                else:
                    new_dict[key] = set(value.keys())
            else:
                new_dict[key] = objects_to_set(value)
        return new_dict
    else:
        return d
    
def convert_sets_to_lists(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_sets_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_sets_to_lists(i) for i in obj]
    else:
        return obj

def transform_dictionary(original_dict):
    final_dict = remove_location_coordinates(original_dict)
    final_dict = objects_to_set(final_dict)
    return final_dict

if __name__ == '__main__':
    file_path = 'map.yaml'
    data = load_yaml_file(file_path)
    transformed_dict = transform_dictionary(data)
    print(transformed_dict)
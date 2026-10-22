import json

def convert_json_format(input_json):
    output_json = {
        "buildings": {}
    }

    building_name = "metrotech-1"
    floor_name = "basement-1"
    room_name = "room-1"

    output_json["buildings"][building_name] = {
        "floors": {
            floor_name: {
                "rooms": {
                    room_name: {
                        "location": [],
                        "objects": {}
                    }
                }
            }
        }
    }

    # Processing objects
    for obj in input_json["metrotech"]["basement"]["map_object_bboxes"]:
        obj_label = obj["label"]
        instance = obj["instance"]
        loc = [obj["center"]["x"], obj["center"]["y"]]
        output_json["buildings"][building_name]["floors"][floor_name]["rooms"][room_name]["objects"][f"{obj_label}-{instance}"] = {
            "location": loc
        }

    # Assuming the room location is the average of the object locations
    locations = [obj["center"] for obj in input_json["metrotech"]["basement"]["map_object_bboxes"]]
    avg_location = [
        sum(loc["x"] for loc in locations) / len(locations),
        sum(loc["y"] for loc in locations) / len(locations)
    ]
    output_json["buildings"][building_name]["floors"][floor_name]["rooms"][room_name]["location"] = avg_location

    return output_json

def main():
    import glob
    import os
    import shutil
    import time

    shutil.rmtree('./maps')
    os.makedirs('./maps')
    while True:
        json_files = glob.glob(os.path.join('/workspace/jsons', '*.json'))
        if not json_files:
            continue
        latest_file = sorted(json_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))[-1]

        # Read JSON from file
        with open(latest_file, 'r') as file:
            input_json = json.load(file)

        save_path = os.path.join('./maps',str(round(time.time()))+'.json')
        output_json = convert_json_format(input_json)
        with open(save_path, 'w') as json_file:
            json.dump(output_json, json_file, indent=4)
        time.sleep(1)

if __name__ == "__main__":
    main()

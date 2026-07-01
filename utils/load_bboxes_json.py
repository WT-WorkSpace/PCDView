import json
import numpy as np

try:
    from utils.utils import wataprint
except ModuleNotFoundError:
    def wataprint(message, type=None):
        print(message)

def get_anno_from_tanway_json(json_data):
        '''
        **功能描述**: 读取 label 为anno格式
        
        Args:
            json_data: tanway json 读取到数据  

        Returns:
            output = {
            "bboxes": np.array(boxes),
            "className": class_list,
            "confidence": confidence_list,
            "movement_state": movement_state_list,
            "id": id_list,
            "link_id": link_id_list,
            'pitch': pitch_list,
            'numPoints': numPoints_list
        } 
        '''
        boxes = []
        class_list = []
        id_list = []
        link_id_list = []
        confidence_list = []
        movement_state_list = []
        tag_list = []
        numPoints_list = []
        pitch_list = []

        for agent in json_data:
            boxes.append(
                [agent["position3d"]["x"], agent["position3d"]["y"], agent["position3d"]["z"], agent["size3d"]["x"],
                agent["size3d"]["y"], agent["size3d"]["z"], agent["heading"]])
            class_list.append(agent["type"])

            if "ID" in agent:
                id_list.append(agent["ID"])
            else:
                id_list.append(None)
            if 'tag' in agent:
                tag_list.append(dict(agent["tag"]) if isinstance(agent["tag"], dict) else {})
                if 'movement_state' in agent["tag"]:
                    movement_state_list.append(agent["tag"]["movement_state"])
                else:
                    movement_state_list.append(None)

                if 'confidence' in agent["tag"]:
                    confidence_list.append(agent["tag"]["confidence"])
                else:
                    confidence_list.append(None)

                link_id_val = agent["tag"].get("link_id") if "link_id" in agent["tag"] else agent["tag"].get("link_ID")
                link_id_list.append(link_id_val)
            elif 'tags' in agent:
                tag_list.append(dict(agent["tags"]) if isinstance(agent["tags"], dict) else {})
                if 'movement_state' in agent["tags"]:
                    movement_state_list.append(agent["tags"]["movement_state"])
                else:
                    movement_state_list.append(None)

                if 'confidence' in agent["tags"]:
                    confidence_list.append(agent["tags"]["confidence"])
                else:
                    confidence_list.append(None)

                link_id_val = agent["tags"].get("link_id") if "link_id" in agent["tags"] else agent["tags"].get("link_ID")
                link_id_list.append(link_id_val)
            else:
                confidence_list.append(None)
                movement_state_list.append(None)
                link_id_list.append(None)
                tag_list.append({})

            numPoints = [None] * 10
            for key in agent.keys():
                if "numPoints" in key:
                    numPoints[int(key.split("numPoints")[-1])] = agent[key]
            numPoints_list.append(numPoints)
            pitch_list.append(agent["pitch"])

        output = {
            "bboxes": np.array(boxes),
            "className": class_list,
            "confidence": confidence_list,
            "movement_state": movement_state_list,
            "id": id_list,
            "link_id": link_id_list,
            "tag": tag_list,
            'pitch': pitch_list,
            'numPoints': numPoints_list
        }
        return output


def save_bboxes_to_tanway_json(json_path, bbox_infos, original_agents=None):
    """
    将 bbox_infos 保存为 Tanway JSON 格式。
    bbox_infos: list of dict, 每个含 x,y,z,l,w,h,yaw,class_name，及可选 pitch/confidence/movement_state。
    输出结构对齐 Tanway 简化格式：type/position3d/size3d/heading/pitch/tag。
    """
    agents_out = []
    for info in bbox_infos:
        x = float(info.get("x", 0))
        y = float(info.get("y", 0))
        z = float(info.get("z", 0))
        l = float(info.get("l", 1))
        w = float(info.get("w", 1))
        h = float(info.get("h", 1))
        yaw = float(info.get("yaw", 0))
        pitch = float(info.get("pitch", 0))
        class_name = info.get("class_name", "others")
        if "TYPE_" not in class_name and class_name != "others":
            type_str = "TYPE_" + class_name
        else:
            type_str = class_name

        agent = {
            "type": type_str,
            "position3d": {"x": x, "y": y, "z": z},
            "size3d": {"x": l, "y": w, "z": h},
            "heading": yaw,
            "pitch": pitch,
        }
        bid = info.get("id")
        if bid is not None:
            agent["ID"] = bid

        tag = {}
        confidence = info.get("confidence")
        movement_state = info.get("movement_state")
        link_id = info.get("link_id")
        if confidence is not None:
            tag["confidence"] = str(confidence)
        if movement_state is not None:
            tag["movement_state"] = str(movement_state)
        if link_id is not None:
            tag["link_id"] = link_id

        reserved_keys = {
            "x", "y", "z", "l", "w", "h", "yaw", "roll", "pitch",
            "class_name", "id", "link_id", "confidence", "movement_state",
            "arrow_yaw", "numPoints",
        }
        for key, value in info.items():
            if key in reserved_keys:
                continue
            if value is not None:
                if isinstance(key, str) and key.startswith("tag.") and "." in key:
                    tag_key = key.split(".", 1)[1].strip()
                    if tag_key:
                        tag[tag_key] = value
                else:
                    tag[key] = value
        if tag:
            agent["tag"] = tag

        agents_out.append(agent)

    with open(json_path, "w", encoding="UTF-8") as f:
        json.dump(agents_out, f, indent=2, ensure_ascii=False)


def get_anno_from_beisai_json(label_path):
    '''
    **功能描述**: 读取 beisai 平台 训练标签 为anno格式
    
    Args:
        json_data: tanway json 读取到数据  

    Returns:
        output = {
            "bbox": bbox,
            "classname": classname,
            "confidence": confidence_list,
            "movement_state": movement_state_list,
            "id": id_list,
            "link_id": link_id_list,
            }
    '''

    classmap = {
        'Car': 'Car',
        'Van': 'Van',
        'Bus': 'Bus',
        'Truck': 'Truck',
        'Semitrailer': 'Semitrailer',
        'Special_vehicles': 'Special_vehicles',
        'Cycle': 'Cycle',
        'Tricyclist': 'Tricyclist',
        'Pedestrian': 'Pedestrian',
        'Vichcle': 'Car',
        'Animal': 'Animal',
        None: None
    }

    with open(label_path, 'r', encoding='UTF-8') as f:
        beisai_data = json.loads(f.read())

    bbox = []
    classname = []
    confidence_list = []
    movement_state_list = []
    id_list = []
    link_id_list = []

    for i, agent in enumerate(beisai_data[0]["objects"]):
        obj_type = None
        confidence = None
        movement_state = None
        link_id = None
        obj_id = None

        contour = agent["contour"]
        bbox.append([contour['center3D']['x'],
                    contour['center3D']['y'],
                    contour['center3D']['z'],
                    contour['size3D']['x'],
                    contour['size3D']['y'],
                    contour['size3D']['z'],
                    contour['rotation3D']['z']])
        assert "classValues" in agent, "classValues not in " + label_path

        for c_v in agent["classValues"]:
            if c_v["name"] == "label" or c_v["name"] in classmap:
                obj_type = c_v["value"]
            if "name" in c_v and c_v["name"] == "confidence":
                confidence = c_v["value"]
            if "name" in c_v and c_v["name"] == "movement_state":
                movement_state = c_v["value"]
            if "name" in c_v and (c_v["name"] == "link_id" or c_v["name"] == "link_ID"):
                link_id = int(c_v["value"])
            if "name" in c_v and c_v["name"] == "ID":
                obj_id = int(c_v["value"])

        if obj_type is None and link_id is not None:
            obj_type = "Semitrailer"

        if obj_type is None:
            wataprint(f"⚠️ {label_path} 中的第{i}个框中无类别信息,请检查!", type="r")

        if obj_type is not None and obj_type not in classmap:
            wataprint(f"Error: {obj_type} not in classmap in {label_path}!", type="r")
            classname.append(obj_type)
        else:
            classname.append(classmap[obj_type])

        confidence_list.append(confidence)
        movement_state_list.append(movement_state)
        id_list.append(obj_id)
        link_id_list.append(link_id)

    assert len(bbox) == len(classname) == len(confidence_list) == len(movement_state_list) == len(id_list) == len(link_id_list)
    output = {
        "bbox": bbox,
        "classname": classname,
        "confidence": confidence_list,
        "movement_state": movement_state_list,
        "id": id_list,
        "link_id": link_id_list,
        }

    return output

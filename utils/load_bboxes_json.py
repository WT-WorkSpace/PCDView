import json
import numpy as np
from utils.utils import wataprint

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
                if 'movement_state' in agent["tag"]:
                    movement_state_list.append(agent["tag"]["movement_state"])
                else:
                    movement_state_list.append(None)

                if 'confidence' in agent["tag"]:
                    confidence_list.append(agent["tag"]["confidence"])
                else:
                    confidence_list.append(None)

                if 'link_id' in agent["tag"]:
                    link_id_list.append(agent["tag"]["link_id"])
                else:
                    link_id_list.append(None)

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
            'pitch': pitch_list,
            'numPoints': numPoints_list
        }
        return output


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
        confidence = 3
        movement_state = 0
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

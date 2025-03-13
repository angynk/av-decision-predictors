import yaml
import xml.etree.ElementTree as ET
import pedestrian_ontology as ontology
import numpy as np
import csv
from ampligraph.evaluation import train_test_split_no_unseen


class PedOccludedKGV4():

    def __init__(self, settings) :
        self.triples = []
        self.settings = settings
        self.counter_gt = {ontology.PED_OCCLUDED:0, ontology.NONE_PED:0, ontology.PED_NOT_OCCLUDED: 0}
        self.header = ["video","frame","pedScene","zebra","surrounding","numVeh","vehFront","vehFrontLeft","vehFrontRight","vehLeft","vehRight"]
        self.csv = []
        self.load_kg()
        self.split_data()

    def load_kg(self):
        self.counter = {ontology.PED_OCCLUDED:{ontology.VEH_ACCELERATING:0, ontology.VEH_DECELERATING:0, ontology.VEH_STOPPED:0,
                                                ontology.VEH_MOVING:0, ontology.VEH_LIGHTS_OFF:0, ontology.VEH_LIGHTS_ON:0,
                                                  ontology.VEH_NEAR:0, ontology.VEH_MIDDLE:0, ontology.VEH_FAR:0,
                                                  ontology.VEGETATION:0, ontology.CLEAR:0, ontology.ZEBRA:0, ontology.NOT_ZEBRA:0},
                   ontology.NONE_PED:{ontology.VEH_ACCELERATING:0, ontology.VEH_DECELERATING:0, ontology.VEH_STOPPED:0,
                                       ontology.VEH_MOVING:0, ontology.VEH_LIGHTS_OFF:0, ontology.VEH_LIGHTS_ON:0,
                                         ontology.VEH_NEAR:0, ontology.VEH_MIDDLE:0, ontology.VEH_FAR:0,
                                                  ontology.VEGETATION:0, ontology.CLEAR:0, ontology.ZEBRA:0, ontology.NOT_ZEBRA:0},
                   }
        self.counter_combination = {ontology.PED_OCCLUDED:{
            ontology.VEH_ACCELERATING+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_NEAR : 0,
            ontology.VEH_ACCELERATING+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_MIDDLE : 0,
            ontology.VEH_ACCELERATING+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_FAR : 0,
            ontology.VEH_ACCELERATING+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_NEAR : 0,
            ontology.VEH_ACCELERATING+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_MIDDLE : 0,
            ontology.VEH_ACCELERATING+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_FAR : 0,
            ontology.VEH_DECELERATING+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_NEAR : 0,
            ontology.VEH_DECELERATING+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_MIDDLE : 0,
            ontology.VEH_DECELERATING+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_FAR : 0,
            ontology.VEH_DECELERATING+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_NEAR : 0,
            ontology.VEH_DECELERATING+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_MIDDLE : 0,
            ontology.VEH_DECELERATING+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_FAR : 0,
            ontology.VEH_STOPPED+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_NEAR : 0,
            ontology.VEH_STOPPED+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_MIDDLE : 0,
            ontology.VEH_STOPPED+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_FAR : 0,
            ontology.VEH_STOPPED+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_NEAR : 0,
            ontology.VEH_STOPPED+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_MIDDLE : 0,
            ontology.VEH_STOPPED+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_FAR : 0,
            ontology.VEH_MOVING+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_NEAR : 0,
            ontology.VEH_MOVING+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_MIDDLE : 0,
            ontology.VEH_MOVING+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_FAR : 0,
            ontology.VEH_MOVING+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_NEAR : 0,
            ontology.VEH_MOVING+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_MIDDLE : 0,
            ontology.VEH_MOVING+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_FAR : 0
        },
        ontology.NONE_PED: {
            ontology.VEH_ACCELERATING+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_NEAR : 0,
            ontology.VEH_ACCELERATING+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_MIDDLE : 0,
            ontology.VEH_ACCELERATING+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_FAR : 0,
            ontology.VEH_ACCELERATING+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_NEAR : 0,
            ontology.VEH_ACCELERATING+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_MIDDLE : 0,
            ontology.VEH_ACCELERATING+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_FAR : 0,
            ontology.VEH_DECELERATING+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_NEAR : 0,
            ontology.VEH_DECELERATING+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_MIDDLE : 0,
            ontology.VEH_DECELERATING+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_FAR : 0,
            ontology.VEH_DECELERATING+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_NEAR : 0,
            ontology.VEH_DECELERATING+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_MIDDLE : 0,
            ontology.VEH_DECELERATING+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_FAR : 0,
            ontology.VEH_STOPPED+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_NEAR : 0,
            ontology.VEH_STOPPED+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_MIDDLE : 0,
            ontology.VEH_STOPPED+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_FAR : 0,
            ontology.VEH_STOPPED+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_NEAR : 0,
            ontology.VEH_STOPPED+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_MIDDLE : 0,
            ontology.VEH_STOPPED+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_FAR : 0,
            ontology.VEH_MOVING+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_NEAR : 0,
            ontology.VEH_MOVING+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_MIDDLE : 0,
            ontology.VEH_MOVING+"-"+ontology.VEH_LIGHTS_OFF+"-"+ontology.VEH_FAR : 0,
            ontology.VEH_MOVING+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_NEAR : 0,
            ontology.VEH_MOVING+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_MIDDLE : 0,
            ontology.VEH_MOVING+"-"+ontology.VEH_LIGHTS_ON+"-"+ontology.VEH_FAR : 0
        }}
        for video in self.settings['videos_train']:
            self.load_road_scene(video)
        print(self.counter)
        print(self.counter_combination)
        if self.settings["extracted_csv"]:
            with open('outputs/database.csv', 'w', encoding='UTF8') as h:
                writer = csv.writer(h)
                # write the header
                writer.writerow(self.header)
                # write the data
                writer.writerows(self.csv)
            h.close()
        
    def load_road_scene (self, scene):
        self.data_scene = ["video","frame","pedScene","zebra","surrounding","numVeh","vehFront","vehFrontLeft","vehFrontRight","vehLeft","vehRight"]
        machine = self.settings['machine']
        scene_annotations = self.settings[machine]['annotations_path']+ "annotations_"+scene+".xml"
        self.scene_data = ET.parse(scene_annotations).getroot()
        road_scene_id, lanes_number, zebra_crossing, surroundings = self.load_context_road_scene()
        scene_frames = self.scene_data.findall("road_scene/scene_frame")
        self.road_scenes = []
        
        for scene_frame in scene_frames:
            data_scene = self.load_road_scene_frame(road_scene_id, scene_frame)
            self.csv.append([scene, data_scene['frame'],data_scene["pedScene"], zebra_crossing, surroundings, data_scene["numVeh"],
                                    data_scene["VehFront"], data_scene["VehFrontLeft"], data_scene["VehFrontRight"],
                                    data_scene["VehLeft"], data_scene["VehRight"]])
            '''if data_scene["pedScene"] != ontology.PED_NOT_OCCLUDED:
                self.counter[data_scene["pedScene"]][surroundings] = self.counter[data_scene["pedScene"]][surroundings] + 1
                self.counter[data_scene["pedScene"]][zebra_crossing] = self.counter[data_scene["pedScene"]][zebra_crossing] + 1'''
        self.load_temporal_relation()
        

    
    def load_temporal_relation(self):
        prev = self.road_scenes[0]
        for i in range(len(self.road_scenes)):
            next = self.road_scenes[i]
            if i != 0:
                self.triples.append((prev, ontology.PREVIOUS, next ))
                self.triples.append((next, ontology.NEXT, prev ))
            prev = next

            


    def load_context_road_scene(self):
        road_scene_id = self.scene_data.find("road_scene").attrib["id"]
        lanes_number = ontology.get_lanes(self.scene_data.find("road_scene/scene_context/lanes").text)
        zebra_crossing = ontology.get_zebra_crossing(self.scene_data.find("road_scene/scene_context/zebra_crossing").text)
        surroundings = self.scene_data.find("road_scene/scene_context/surroundings").text
        self.triples.append((ontology.ROAD_SCENE, ontology.HAS_CHILD, road_scene_id ))
        self.triples.append((road_scene_id, ontology.HAS, lanes_number ))
        self.triples.append((road_scene_id, ontology.THERE_IS, zebra_crossing ))
        self.triples.append((road_scene_id, ontology.SURROUNDING, surroundings ))

        if zebra_crossing == ontology.ZEBRA:
            zebra_crossing = 1
        else:
            zebra_crossing = 0
        
        if surroundings == ontology.VEGETATION:
            surroundings = 1
        else:
            surroundings = 0
        return road_scene_id, lanes_number, zebra_crossing, surroundings
    
    def load_road_scene_frame(self, road_scene_id, scene_frame):
    
        scene_frame_id = road_scene_id+"_"+scene_frame.attrib["id"]
        pedestrian_oclussion = scene_frame.find("pedestrians_scene").text
        data_scene = {"frame":scene_frame.attrib["id"],
                      "pedScene": pedestrian_oclussion ,
                      "numVeh": 0 ,"VehFront": "" , "VehFrontLeft": "" ,
                      "VehFrontRight": "" , "VehLeft": "", "VehRight": ""}
        
        self.counter_gt[pedestrian_oclussion] = self.counter_gt[pedestrian_oclussion] + 1

        if pedestrian_oclussion != ontology.PED_NOT_OCCLUDED:
            self.triples.append((scene_frame_id, ontology.INSTANCE_OF, road_scene_id ))
            self.triples.append((scene_frame_id, ontology.CONTAINS, pedestrian_oclussion ))
            self.road_scenes.append(scene_frame_id)

            pedestrians = scene_frame.findall("pedestrians/pedestrian")
            for ped in pedestrians:
                ped_id = ped.attrib["id"]
                occlusion_type = ped.find("occlusion").text
                if occlusion_type == "Na":
                    occlusion_type = ontology.NONE_OCCLUSION 
                #crossing_action = ped.find("crossing_action").text
                self.triples.append((ontology.PEDESTRIAN, ontology.HAS_CHILD, ped_id ))
                self.triples.append((ped_id, ontology.OCCLUSION_TYPE, occlusion_type ))
                #self.triples.append((ped_id, ontology.INTENTION, crossing_action ))
                self.triples.append((scene_frame_id, ontology.INCLUDES_PED, ped_id ))

            vehicles = scene_frame.findall("vehicles/vehicle")
            data_scene["numVeh"] = len(vehicles)
            if len(vehicles) > 0:
                for veh in vehicles:
                    veh_id = veh.attrib["id"]
                    veh_label = veh.attrib["label"]
                    state = veh.find("state").text
                    braking_ligths = veh.find("braking_ligths").text
                    distance = veh.find("distance").text
                    self.triples.append((ontology.VEHICLE, ontology.HAS_CHILD, veh_id ))
                    self.triples.append((veh_id, ontology.IS, veh_label ))
                    self.triples.append((veh_id, ontology.STATE, state ))
                    self.triples.append((veh_id, ontology.BRAKING_LIGTHS, braking_ligths ))
                    self.triples.append((veh_id, ontology.DISTANCE, distance ))
                    self.triples.append((scene_frame_id, ontology.INCLUDES_VEH, veh_id ))
                    self.counter[pedestrian_oclussion][state] = self.counter[pedestrian_oclussion][state] + 1
                    self.counter[pedestrian_oclussion][braking_ligths] = self.counter[pedestrian_oclussion][braking_ligths] + 1
                    self.counter[pedestrian_oclussion][distance] = self.counter[pedestrian_oclussion][distance] + 1
                    combination = state+"-"+braking_ligths+"-"+distance
                    self.counter_combination[pedestrian_oclussion][combination] = self.counter_combination[pedestrian_oclussion][combination] + 1

                    data_scene[veh_label] = self.get_value_state(state) 
            else:
                self.triples.append((scene_frame_id, ontology.INCLUDES_VEH, ontology.ISOLATED ))
            
        return data_scene

        
    def get_value_state(self, state):
        if state == ontology.VEH_STOPPED:
            return 0
        if state == ontology.VEH_MOVING:
            return 1
        if state == ontology.VEH_ACCELERATING:
            return 2
        return -1
    
    def split_data(self):
        self.X_train, self.X_valid = train_test_split_no_unseen(np.array(self.triples), test_size=3000)
        print('Train set size: ', self.X_train.shape)
        print('Test set size: ', self.X_valid.shape)
        print(self.counter_gt)



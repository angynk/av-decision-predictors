from ampligraph.utils import restore_model
import pedestrian_ontology as ontology
import numpy as np
import xml.etree.ElementTree as ET
from pedestrian_occluded_KG import PedOccludedKGV4
import csv
from ultralytics import YOLO
import cv2


def load_base_predictions_v4(bayes):
    base_probabilities = {
        ontology.HYPHOTESIS_OCCLUDED_PED : bayes.evaluate_triple(np.array([[ontology.ROAD_SCENE, ontology.CONTAINS, ontology.HYPHOTESIS_OCCLUDED_PED ]])),
        ontology.HYPHOTESIS_NONE_PED : bayes.evaluate_triple(np.array([[ontology.ROAD_SCENE, ontology.CONTAINS, ontology.HYPHOTESIS_NONE_PED ]])),

        ontology.EVIDENCE:{
            ontology.ZEBRA : bayes.evaluate_triple(np.array([[ontology.ROAD_SCENE, ontology.THERE_IS, ontology.ZEBRA ]])),
            ontology.NOT_ZEBRA : bayes.evaluate_triple(np.array([[ontology.ROAD_SCENE, ontology.THERE_IS, ontology.NOT_ZEBRA ]])),
            #ontology.VEGETATION : bayes.evaluate_triple(np.array([[ontology.ROAD_SCENE, ontology.SURROUNDING, ontology.VEGETATION ]])),
            #ontology.CLEAR : bayes.evaluate_triple(np.array([[ontology.ROAD_SCENE, ontology.SURROUNDING, ontology.CLEAR ]])),
            ontology.TWO_LANES : bayes.evaluate_triple(np.array([[ontology.ROAD_SCENE, ontology.HAS, ontology.TWO_LANES ]])),
            ontology.FOUR_LANES : bayes.evaluate_triple(np.array([[ontology.ROAD_SCENE, ontology.HAS, ontology.FOUR_LANES ]])),
            
            ontology.VEH_MOVING : bayes.evaluate_triple(np.array([[ontology.ROAD_SCENE, ontology.STATE, ontology.VEH_MOVING ]])),
            ontology.VEH_STOPPED : bayes.evaluate_triple(np.array([[ontology.ROAD_SCENE, ontology.STATE, ontology.VEH_STOPPED ]])),
            ontology.VEH_ACCELERATING : bayes.evaluate_triple(np.array([[ontology.ROAD_SCENE, ontology.STATE, ontology.VEH_ACCELERATING ]])),
            ontology.VEH_DECELERATING : bayes.evaluate_triple(np.array([[ontology.ROAD_SCENE, ontology.STATE, ontology.VEH_DECELERATING ]])),
            ontology.VEH_LIGHTS_ON : bayes.evaluate_triple(np.array([[ontology.ROAD_SCENE, ontology.BRAKING_LIGTHS, ontology.VEH_LIGHTS_ON ]])),
            ontology.VEH_LIGHTS_OFF : bayes.evaluate_triple(np.array([[ontology.ROAD_SCENE, ontology.BRAKING_LIGTHS, ontology.VEH_LIGHTS_OFF ]])),
            ontology.VEH_NEAR : bayes.evaluate_triple(np.array([[ontology.ROAD_SCENE, ontology.DISTANCE, ontology.VEH_NEAR ]])),
            ontology.VEH_MIDDLE : bayes.evaluate_triple(np.array([[ontology.ROAD_SCENE, ontology.DISTANCE, ontology.VEH_MIDDLE ]])),
            ontology.VEH_FAR : bayes.evaluate_triple(np.array([[ontology.ROAD_SCENE, ontology.DISTANCE, ontology.VEH_FAR ]])),

            ontology.ISOLATED : bayes.evaluate_triple(np.array([[ontology.ROAD_SCENE, ontology.INCLUDES_VEH, ontology.ISOLATED ]])),
            ontology.VEH_FRONT_LEFT : bayes.evaluate_triple(np.array([[ontology.ROAD_SCENE, ontology.IS, ontology.VEH_FRONT_LEFT ]])),
            ontology.VEH_FRONT_RIGHT : bayes.evaluate_triple(np.array([[ontology.ROAD_SCENE, ontology.IS, ontology.VEH_FRONT_RIGHT ]])),
            ontology.VEH_LEFT : bayes.evaluate_triple(np.array([[ontology.ROAD_SCENE, ontology.IS, ontology.VEH_LEFT ]])),
            ontology.VEH_RIGHT : bayes.evaluate_triple(np.array([[ontology.ROAD_SCENE, ontology.IS, ontology.VEH_RIGHT ]]))
        },

        ontology.EVIDENCE_HYPHOTESIS_OCCLUDED_PED : {
            ontology.ZEBRA : bayes.evaluate_triple(np.array([[ontology.ZEBRA, ontology.CONTAINS, ontology.HYPHOTESIS_OCCLUDED_PED ]])),
            ontology.NOT_ZEBRA : bayes.evaluate_triple(np.array([[ontology.NOT_ZEBRA, ontology.CONTAINS, ontology.HYPHOTESIS_OCCLUDED_PED ]])),
            #ontology.VEGETATION : bayes.evaluate_triple(np.array([[ontology.VEGETATION, ontology.CONTAINS,ontology.HYPHOTESIS_OCCLUDED_PED  ]])),
            #ontology.CLEAR : bayes.evaluate_triple(np.array([[ontology.CLEAR , ontology.CONTAINS, ontology.HYPHOTESIS_OCCLUDED_PED]])),
            ontology.TWO_LANES : bayes.evaluate_triple(np.array([[ontology.TWO_LANES, ontology.HAS, ontology.HYPHOTESIS_OCCLUDED_PED ]])),
            ontology.FOUR_LANES : bayes.evaluate_triple(np.array([[ontology.FOUR_LANES, ontology.HAS, ontology.HYPHOTESIS_OCCLUDED_PED ]])),
            
            ontology.VEH_MOVING : bayes.evaluate_triple(np.array([[ontology.VEH_MOVING, ontology.CONTAINS, ontology.HYPHOTESIS_OCCLUDED_PED ]])),
            ontology.VEH_STOPPED : bayes.evaluate_triple(np.array([[ontology.VEH_STOPPED, ontology.CONTAINS, ontology.HYPHOTESIS_OCCLUDED_PED ]])),
            ontology.VEH_ACCELERATING : bayes.evaluate_triple(np.array([[ontology.VEH_ACCELERATING, ontology.CONTAINS, ontology.HYPHOTESIS_OCCLUDED_PED ]])),
            ontology.VEH_DECELERATING : bayes.evaluate_triple(np.array([[ ontology.VEH_DECELERATING, ontology.CONTAINS, ontology.HYPHOTESIS_OCCLUDED_PED]])),
            ontology.VEH_LIGHTS_ON : bayes.evaluate_triple(np.array([[ontology.VEH_LIGHTS_ON, ontology.CONTAINS, ontology.HYPHOTESIS_OCCLUDED_PED ]])),
            ontology.VEH_LIGHTS_OFF : bayes.evaluate_triple(np.array([[ontology.VEH_LIGHTS_OFF, ontology.CONTAINS, ontology.HYPHOTESIS_OCCLUDED_PED ]])),
            ontology.VEH_NEAR : bayes.evaluate_triple(np.array([[ontology.VEH_NEAR, ontology.CONTAINS, ontology.HYPHOTESIS_OCCLUDED_PED ]])),
            ontology.VEH_MIDDLE : bayes.evaluate_triple(np.array([[ontology.VEH_MIDDLE, ontology.CONTAINS, ontology.HYPHOTESIS_OCCLUDED_PED ]])),
            ontology.VEH_FAR : bayes.evaluate_triple(np.array([[ontology.VEH_FAR, ontology.CONTAINS, ontology.HYPHOTESIS_OCCLUDED_PED ]])),

            ontology.ISOLATED : bayes.evaluate_triple(np.array([[ontology.ISOLATED, ontology.CONTAINS, ontology.HYPHOTESIS_OCCLUDED_PED ]])),
            ontology.VEH_FRONT_LEFT : bayes.evaluate_triple(np.array([[ontology.VEH_FRONT_LEFT, ontology.INCLUDES_VEH, ontology.HYPHOTESIS_OCCLUDED_PED ]])),
            ontology.VEH_FRONT_RIGHT : bayes.evaluate_triple(np.array([[ontology.VEH_FRONT_RIGHT, ontology.INCLUDES_VEH, ontology.HYPHOTESIS_OCCLUDED_PED ]])),
            ontology.VEH_LEFT : bayes.evaluate_triple(np.array([[ontology.VEH_LEFT, ontology.INCLUDES_VEH, ontology.HYPHOTESIS_OCCLUDED_PED ]])),
            ontology.VEH_RIGHT : bayes.evaluate_triple(np.array([[ontology.VEH_RIGHT, ontology.INCLUDES_VEH, ontology.HYPHOTESIS_OCCLUDED_PED ]]))

        } ,
        ontology.EVIDENCE_HYPHOTESIS_NONE_PED : {
            ontology.ZEBRA : bayes.evaluate_triple(np.array([[ontology.ZEBRA, ontology.CONTAINS, ontology.HYPHOTESIS_NONE_PED ]])),
            ontology.NOT_ZEBRA : bayes.evaluate_triple(np.array([[ontology.NOT_ZEBRA, ontology.CONTAINS, ontology.HYPHOTESIS_NONE_PED ]])),
            #ontology.VEGETATION : bayes.evaluate_triple(np.array([[ontology.VEGETATION, ontology.CONTAINS,ontology.HYPHOTESIS_NONE_PED  ]])),
            #ontology.CLEAR : bayes.evaluate_triple(np.array([[ontology.CLEAR , ontology.CONTAINS, ontology.HYPHOTESIS_NONE_PED]])),
            ontology.TWO_LANES : bayes.evaluate_triple(np.array([[ontology.TWO_LANES, ontology.HAS, ontology.HYPHOTESIS_NONE_PED ]])),
            ontology.FOUR_LANES : bayes.evaluate_triple(np.array([[ontology.FOUR_LANES, ontology.HAS, ontology.HYPHOTESIS_NONE_PED ]])),
            
            ontology.VEH_MOVING : bayes.evaluate_triple(np.array([[ontology.VEH_MOVING, ontology.CONTAINS, ontology.HYPHOTESIS_NONE_PED ]])),
            ontology.VEH_STOPPED : bayes.evaluate_triple(np.array([[ontology.VEH_STOPPED, ontology.CONTAINS, ontology.HYPHOTESIS_NONE_PED ]])),
            ontology.VEH_ACCELERATING : bayes.evaluate_triple(np.array([[ontology.VEH_ACCELERATING, ontology.CONTAINS, ontology.HYPHOTESIS_NONE_PED ]])),
            ontology.VEH_DECELERATING : bayes.evaluate_triple(np.array([[ ontology.VEH_DECELERATING, ontology.CONTAINS, ontology.HYPHOTESIS_NONE_PED]])),
            ontology.VEH_LIGHTS_ON : bayes.evaluate_triple(np.array([[ontology.VEH_LIGHTS_ON, ontology.CONTAINS, ontology.HYPHOTESIS_NONE_PED ]])),
            ontology.VEH_LIGHTS_OFF : bayes.evaluate_triple(np.array([[ontology.VEH_LIGHTS_OFF, ontology.CONTAINS, ontology.HYPHOTESIS_NONE_PED ]])),
            ontology.VEH_NEAR : bayes.evaluate_triple(np.array([[ontology.VEH_NEAR, ontology.CONTAINS, ontology.HYPHOTESIS_NONE_PED ]])),
            ontology.VEH_MIDDLE : bayes.evaluate_triple(np.array([[ontology.VEH_MIDDLE, ontology.CONTAINS, ontology.HYPHOTESIS_NONE_PED ]])),
            ontology.VEH_FAR : bayes.evaluate_triple(np.array([[ontology.VEH_FAR, ontology.CONTAINS, ontology.HYPHOTESIS_NONE_PED ]])),

            ontology.ISOLATED : bayes.evaluate_triple(np.array([[ontology.ISOLATED, ontology.CONTAINS, ontology.HYPHOTESIS_NONE_PED ]])),
            ontology.VEH_FRONT_LEFT : bayes.evaluate_triple(np.array([[ontology.VEH_FRONT_LEFT, ontology.INCLUDES_VEH, ontology.HYPHOTESIS_NONE_PED ]])),
            ontology.VEH_FRONT_RIGHT : bayes.evaluate_triple(np.array([[ontology.VEH_FRONT_RIGHT, ontology.INCLUDES_VEH, ontology.HYPHOTESIS_NONE_PED ]])),
            ontology.VEH_LEFT : bayes.evaluate_triple(np.array([[ontology.VEH_LEFT, ontology.INCLUDES_VEH, ontology.HYPHOTESIS_NONE_PED ]])),
            ontology.VEH_RIGHT : bayes.evaluate_triple(np.array([[ontology.VEH_RIGHT, ontology.INCLUDES_VEH, ontology.HYPHOTESIS_NONE_PED ]]))

        }
    }
    return base_probabilities

class OcclusionPredictor():

    def __init__(self, settings) :
        self.settings = settings
        self.load_model()
        self.load_base_predictions()

    def load_model(self):
        
        kg = PedOccludedKGV4(self.settings)
        
        self.model = restore_model(model_name_path="models/"+self.settings['model_name'])
        
        self.model.calibrate(kg.X_train, 
                X_neg=None, 
                positive_base_rate=self.settings['positive_base_rate'], 
                batch_size=self.settings['batch'], 
                epochs=self.settings['epochs_calibration'], 
                verbose=False)
        
        self.yolo_model = YOLO("yolov8n.pt") 

    
    def load_base_predictions(self):
        #file_results_frame = 'outputs/map.csv'
        #f_1 = open(file_results_frame, 'w', encoding='UTF8')
        #writer_results_frame = csv.writer(f_1)
        #writer_results_frame.writerow(["Feature","Ped Occluded", "Ped Not Occluded","None Ped"])
        self.base_probs = load_base_predictions_v4(self)
        
        #f_1.close()

    def evaluate_triple(self, triple):
        triple_score = self.model.predict_proba(triple) 
        return triple_score
    
    def softmax(self,x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0) # only difference
    
    def get_numerical_value_occlusion(self, value, pred):
        if pred == False:
            if value is not None:
                value = value.text
            else:
                return -1
        if value == ontology.PED_OCCLUDED :
            return 0
        elif value == ontology.NONE_PED:
            return 1
        else:
            return 2
        
    def include_veh_v4(self, evidence, eviHyp_ped_occluded, eviHyp_none_ped, vehicle,vehicle_data):
        state =vehicle[0]
        braking_lights = vehicle[1]
        distance = vehicle[2]
        position = vehicle[3]
        #State
        evidence.append(self.base_probs[ontology.EVIDENCE][state])
        eviHyp_ped_occluded.append(self.base_probs[ontology.EVIDENCE_HYPHOTESIS_OCCLUDED_PED][state])
        eviHyp_none_ped.append(self.base_probs[ontology.EVIDENCE_HYPHOTESIS_NONE_PED][state])
        # Braking Ligths
        evidence.append(self.base_probs[ontology.EVIDENCE][braking_lights])
        eviHyp_ped_occluded.append(self.base_probs[ontology.EVIDENCE_HYPHOTESIS_OCCLUDED_PED][braking_lights])
        eviHyp_none_ped.append(self.base_probs[ontology.EVIDENCE_HYPHOTESIS_NONE_PED][braking_lights])
        #Distance to ego vehicle
        evidence.append(self.base_probs[ontology.EVIDENCE][distance])
        eviHyp_ped_occluded.append(self.base_probs[ontology.EVIDENCE_HYPHOTESIS_OCCLUDED_PED][distance])
        eviHyp_none_ped.append(self.base_probs[ontology.EVIDENCE_HYPHOTESIS_NONE_PED][distance])
        # Vehicle Position
        #evidence.append(self.base_probs[ontology.EVIDENCE][position])
        #eviHyp_ped_occluded.append(self.base_probs[ontology.EVIDENCE_HYPHOTESIS_OCCLUDED_PED][position])
        #eviHyp_none_ped.append(self.base_probs[ontology.EVIDENCE_HYPHOTESIS_NONE_PED][position])
        vehicle_data[0] = vehicle_data[0]+","+state
        vehicle_data[1] = vehicle_data[1]+","+braking_lights
        vehicle_data[2] = vehicle_data[2]+","+distance
        return evidence, eviHyp_ped_occluded, eviHyp_none_ped, vehicle_data
    

    def predict_scene(self, zebra, lanes, surroundings, vehicles):
        hyphotesis_ped_occluded_score = self.base_probs[ontology.HYPHOTESIS_OCCLUDED_PED]
        hyphotesis_none_ped_score = self.base_probs[ontology.HYPHOTESIS_NONE_PED]
        evidence = []
        eviHyp_ped_occluded = []
        eviHyp_none_ped = []
        vehicle_data = ["","",""]

        #CONTEXTUAL DATA
        #Zebra crossing
        evidence.append(self.base_probs[ontology.EVIDENCE][zebra])
        eviHyp_ped_occluded.append(self.base_probs[ontology.EVIDENCE_HYPHOTESIS_OCCLUDED_PED][zebra])
        eviHyp_none_ped.append(self.base_probs[ontology.EVIDENCE_HYPHOTESIS_NONE_PED][zebra])
        #Lanes number
        #evidence.append(self.base_probs[ontology.EVIDENCE][lanes])
        #eviHyp_ped_occluded.append(self.base_probs[ontology.EVIDENCE_HYPHOTESIS_OCCLUDED_PED][lanes])
        #eviHyp_none_ped.append(self.base_probs[ontology.EVIDENCE_HYPHOTESIS_NONE_PED][lanes])
        #Surroundings
        #evidence.append(self.base_probs[ontology.EVIDENCE][surroundings])
        #eviHyp_ped_occluded.append(self.base_probs[ontology.EVIDENCE_HYPHOTESIS_OCCLUDED_PED][surroundings])
        #eviHyp_none_ped.append(self.base_probs[ontology.EVIDENCE_HYPHOTESIS_NONE_PED][surroundings])

        #VEHICLES
        if len(vehicles) >0:
            for vehicle in vehicles:
                evidence, eviHyp_ped_occluded, eviHyp_none_ped, vehicle_data = self.include_veh_v4(evidence, eviHyp_ped_occluded, eviHyp_none_ped, vehicle, vehicle_data)
        else:
            evidence.append(self.base_probs[ontology.EVIDENCE][ontology.ISOLATED])
            eviHyp_ped_occluded.append(self.base_probs[ontology.EVIDENCE_HYPHOTESIS_OCCLUDED_PED][ontology.ISOLATED])
            eviHyp_none_ped.append(self.base_probs[ontology.EVIDENCE_HYPHOTESIS_NONE_PED][ontology.ISOLATED])

        #EVIDENCE 
        evidence_score = np.prod(evidence)

        #EVIDENCE|HYPHOTESIS
        eviHyp_ped_occluded_score = np.prod(eviHyp_ped_occluded)
        eviHyp_none_ped_score = np.prod(eviHyp_none_ped)

        probability_ped_occluded = (hyphotesis_ped_occluded_score * eviHyp_ped_occluded_score)/evidence_score
        probability_none_ped = (hyphotesis_none_ped_score * eviHyp_none_ped_score)/evidence_score

        probabilities = [probability_ped_occluded[0], probability_none_ped[0]]
        #print('Results')
        probabilities = self.softmax(probabilities)
        prediction = self.get_prediction(probabilities)

        return prediction, probabilities, vehicle_data

    
    def get_prediction(self, probabilities):
        if probabilities[0] >= probabilities[1]:
            return ontology.PED_OCCLUDED
        else:
            return ontology.NONE_PED






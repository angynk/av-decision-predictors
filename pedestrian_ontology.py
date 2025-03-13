ROAD_SCENE = "RoadScene"
PEDESTRIAN = "Pedestrian"
VEHICLE = "Vehicle"
HAS_CHILD = "HasChild"
HAS = "Has"
LANES = "Lanes"
CONTAINS = "Contains"
SURROUNDING = "Surrounding"
ZEBRA = "ZebraCrossing"
NOT_ZEBRA = "NotZebraCrossing"
INCLUDES_PED = "IncludesPedestrian"
INTENTION = "Intention"
OCCLUSION_TYPE = "OcclusionType"
INCLUDES_VEH = "IncludesVehicle"
BRAKING_LIGTHS = "BrakingLigths"
STATE = "State"
DISTANCE = "Distance"
THERE_IS = "ThereIs"
NEXT = "Next"
PREVIOUS = "Previous"
INSTANCE_OF = "InstanceOf"
IS = "Is"
HYPHOTESIS_OCCLUDED_PED = "PedOccluded"
HYPHOTESIS_NOT_OCCLUDED_PED = "PedNotOccluded"
HYPHOTESIS_NONE_PED = "NonePedestrian"
EVIDENCE = "Evidence"
EVIDENCE_HYPHOTESIS_OCCLUDED_PED = "Evidence_hyph_occluded_ped"
EVIDENCE_HYPHOTESIS_NOT_OCCLUDED_PED = "Evidence_hyph_not_occluded_ped"
EVIDENCE_HYPHOTESIS_NONE_PED = "Evidence_hyph_none_ped"
ISOLATED = "Isolated"

VEGETATION = "Vegetation"
CLEAR = "Clear"
TWO_LANES = "2Lanes"
FOUR_LANES = "4Lanes"
FULL_OCCLUSION = "Full"
PARTIAL_OCCLUSION = "Partial"
NONE_OCCLUSION = "None"
VEH_MOVING = "ContiniousMovement"
VEH_STOPPED = "Stopped"
VEH_ACCELERATING = "Acelerating"
VEH_DECELERATING = "Decelerating" 
VEH_LIGHTS_ON = "on"
VEH_LIGHTS_OFF = "off"
VEH_NEAR = "NearToEgoVeh"
VEH_MIDDLE = "MiddleDisToEgoVeh"
VEH_FAR = "FarToEgoVeh"
VEH_FRONT_LEFT = "VehFrontLeft"
VEH_FRONT_RIGHT = "VehFrontRight"
VEH_LEFT = "VehLeft"
VEH_RIGHT = "VehRight"

PED_OCCLUDED = "PedOccluded"
PED_NOT_OCCLUDED = "PedNotOccluded"
NONE_PED = "NonePedestrian"

HAS_INSTANCE = "HasInstance"
INCLUDES = "Includes"
NOT_INCLUDES = "NotIncludes"
HAS_CHILD_VEH = "HasChildVeh"
VALUE_HAS_CHILD = 1
VALUE_HAS_INSTANCE = 1
VALUE_NEXT = 0.7
VALUE_PREVIOUS = 0.7
VALUE_CONTAINS = 1
VALUE_THERE_IS = 0.6
VALUE_SURROUNDING = 0.2
VALUE_INCLUDES = 1
VALUE_NOT_INCLUDES = 1
VALUE_HAS_CHILD_VEH = 1
VALUE_DISTANCE = 0.95
VALUE_STATE = 0.7
VALUE_BRAKING_LIGHTS = 0.5
VALUE_IS = 0.3





def get_lanes(lanes_number):
    return lanes_number + LANES

def get_zebra_crossing(zebra):
    zebra_crossing = ZEBRA if zebra == "True" else NOT_ZEBRA
    return zebra_crossing
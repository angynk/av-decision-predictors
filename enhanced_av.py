#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import numpy.random as random
import re
import sys
import time
import weakref
import yaml
from yaml.loader import SafeLoader

from occlusion_predictor import OcclusionPredictor

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent  # pylint: disable=import-error
from agents.navigation.global_route_planner import GlobalRoutePlanner


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


# ==============================================================================
# -- Custom Agent Pedestrian Occluded ------------------------------------------
# ==============================================================================
class CustomAgent(BasicAgent):
    def __init__(self, vehicle, settings,model_conf, tm):
        self.ego_vehicle = vehicle
        self.settings = settings
        self.reaction_time = self.settings['ENHANCED_VEH']['REACTION_TIME']  # AV reaction time (in seconds)
        self.safe_deceleration = self.settings['ENHANCED_VEH']['SAFE_DECELERATING']   # Maximum comfortable deceleration (m/s²)
        self.safe_distance_margin = self.settings['ENHANCED_VEH']['SAFE_DISTANCE_MARGIN']   # Extra safe stopping distance (meters)
        self.k_risk = self.settings['ENHANCED_VEH']['OCCLUSION_SENSITIVY']   # Occlusion sensitivity factor
        self.occlusion_predictor = OcclusionPredictor(model_conf)
        self.scenario = self.settings['SCENARIO'] 
        self.prev_speed = 0.0  # Previous speed in m/s
        self.prev_time = time.time()  # Previous timestamp
        self.prob_occlusion = []
        self.deceleration = None
        #self.tm= tm
        self.car_decisions = []
        self._ignore_traffic_lights = True
        self._ignore_stop_signs = True

    def is_light_red(self, vehicle):
        """ Override this function to always return False. """
        return False  # This makes the agent ignore red lights

    def _vehicle_obstacle_detected(self, vehicle_list):
        """ Override this method to ignore stop signs and crosswalks. """
        return None

    def get_vehicle_speed(self, vehicle):
        velocity = vehicle.get_velocity()
        #return 3.6*math.sqrt(velocity.x**2 + velocity.y**2 +velocity.z**2)
        return 3.6*math.sqrt(velocity.x**2 + velocity.y**2 +velocity.z**2)
    
    def get_acceleration(self, current_speed):
        """Compute acceleration based on speed change"""
        current_time = time.time()
        dt = current_time - self.prev_time  # Time difference

        if dt == 0:
            return 0.0  # Avoid division by zero

        acceleration = (current_speed - self.prev_speed) / dt

        # Update previous values for next computation
        self.prev_speed = current_speed
        self.prev_time = current_time

        return acceleration
    
    def compute_braking_distance(self, speed): 
        """ Compute braking distance based on physics equations """
        #speed = speed/3.6
        reaction_distance = speed * self.reaction_time
        stopping_distance = (speed ** 2) / (2 * self.safe_deceleration)
        return reaction_distance + stopping_distance
        #return stopping_distance
    
    def adjust_speed_based_on_occlusion(self, speed, distance, vf=0): #vf = 0 means fully stop vehicle -> kinematic equation
        """ Adjust speed dynamically based on pedestrian occlusion probability """
        #return speed * (1 - self.k_risk * occlusion_prob)
        if distance <= 0:
            #raise ValueError("Distance must be greater than 0")
            return -2, speed

        if speed**2 < 2 * abs(vf**2) / distance:
            raise ValueError("Deceleration too high for the given distance")

        # Compute required acceleration
        a = (vf**2 - speed**2) / (2 * distance)
        #a = -7
        #a = max(a, -3)

        # Compute target speed (if needed)
        target_speed = math.sqrt(max(speed**2 + 2 * a * distance, 0))

        return a, target_speed

    def calculate_required_speed(self, initial_speed, stop_distance, max_deceleration=-3.0):
        """
        Calcula la velocidad objetivo para que el vehículo se detenga en una distancia dada.

        :param initial_speed: Velocidad actual en m/s.
        :param stop_distance: Distancia en metros donde debe detenerse.
        :param max_deceleration: Aceleración máxima permitida (negativa).
        :return: Velocidad objetivo en m/s.
        """
        # Evitar divisiones por cero o valores negativos de distancia
        if stop_distance <= 0:
            return 0.0

        # Calcular la aceleración necesaria para detenerse
        required_acceleration = - (initial_speed ** 2) / (2 * stop_distance)

        # Limitar la aceleración dentro del rango permitido
        acceleration = max(required_acceleration, max_deceleration)

        # Calcular la velocidad objetivo
        target_speed = (initial_speed ** 2 + 2 * acceleration * stop_distance) ** 0.5

        return max(target_speed, 0.0) 

    def get_vehicle_state(self, vehicle_occl):
        """Determine if the vehicle is accelerating, decelerating, stopped, or maintaining speed"""
        speed = self.get_vehicle_speed(vehicle_occl)
        acceleration = self.get_acceleration(speed)

        if speed < 0.1:
            return "Stopped" , "off"
        elif acceleration > 0.1:
            return "Acelerating", "off"
        elif acceleration < -0.1:
            return "Decelerating", "on"
        else:
            return "ContiniousMovement", "off"
        
    def get_vehicle_distance_value (self, vehicle_occl):
        distance_to_occl_vehicle = self.ego_vehicle.get_location().distance(vehicle_occl.get_location())
        if  distance_to_occl_vehicle > 0.5 and distance_to_occl_vehicle <= 3.8:
            return 'NearToEgoVeh'
        elif distance_to_occl_vehicle > 3.8 and distance_to_occl_vehicle <= 6:
            return 'MiddleDisToEgoVeh'
        else:
            return 'FarToEgoVeh'
    
    def get_occlusion_prediction(self, vehicle_occl):
        state, braking_ligths = self.get_vehicle_state(vehicle_occl)
        distance = self.get_vehicle_distance_value(vehicle_occl)
        vehicle_occl_data = [state, braking_ligths, distance, self.settings[self.scenario]["OCCL_POSITION"]]
        prediction, probabilities, vehicle_data = self.occlusion_predictor.predict_scene(self.settings[self.scenario]["ZEBRA_CROSSING"], 
                                                                                         self.settings[self.scenario]["LANES"], self.settings[self.scenario]["SURROUNDINGS"], [vehicle_occl_data])
        print("VEHICLE OCCLUDED: "+state+", "+braking_ligths+", "+distance+" ->"+prediction+": "+str(probabilities[0]))
        return probabilities[0]


    def estimate_time_to_collision(self, ego_speed, distance_to_pedestrian_ego_veh, pedestrian_velocity):
        pedestrian_speed = math.sqrt(pedestrian_velocity.x**2 + pedestrian_velocity.y**2 +pedestrian_velocity.z**2)
        relative_speed = ego_speed - pedestrian_speed
        ttc = distance_to_pedestrian_ego_veh / relative_speed
        return ttc


    def run_step(self, pedestrian, vehicle_occl, ped_cross):
        # Get current vehicle speed
        speed = self.get_vehicle_speed(self.ego_vehicle)

        if ped_cross == 3:
            occlusion_prob = 0
            self.ego_vehicle.set_autopilot(True, self.tm.get_port())
        else:
            occlusion_prob = self.get_occlusion_prediction(vehicle_occl)

        # Compute safe stopping distance
        

        # Compute distance to pedestrian
        control = carla.VehicleControl()
        
        target_speed = self.settings[self.scenario]["EGO_SPEED"]
        ttc_anticipation = None
        
        # TAKE CAR DECISION
        if occlusion_prob >= self.settings["ENHANCED_VEH"]["OCCLUSION_SENSITIVY"]:
            self.prob_occlusion.append(1)
        else:
            self.prob_occlusion.append(0)

        if len(self.prob_occlusion) >= 10 and ped_cross != 2:
                if all(x == self.prob_occlusion[-1] for x in self.prob_occlusion[-5:]) and self.prob_occlusion[len(self.prob_occlusion)-1] != 0:
                    ped_cross = 1 

        distance_to_pedestrian = self.ego_vehicle.get_location().distance(pedestrian.get_location())

        #if ped_cross == 1 and speed > 1 and distance_to_pedestrian <= self.settings[self.scenario]["DIS_TO_CONTROL_EGO"]:
        if ped_cross == 1 and speed > 1:    
            braking_distance = self.compute_braking_distance((self.settings[self.scenario]["EGO_SPEED"] + 0.78)/3.6) + 2
            distance_to_occluded_car = self.ego_vehicle.get_location().distance(vehicle_occl.get_location())

            target_deceleration, target_speed = self.adjust_speed_based_on_occlusion((speed+0.78)/3.6, distance_to_pedestrian - 3)
            brake = round(abs(target_deceleration)/8, 2)
            decision = {"decision": "BRAKING", "Explanation": "PEDESTRIAN OCCLUDED", "brake": brake, "throttle": 0.0, "gear": 0, "steer": 0.0, "manual_gear": 1, "auto_pilot": False ,"speed": speed }
            ped_cross = 2
            #self.ego_vehicle.set_autopilot(False, self.tm.get_port())
            ttc_anticipation = self.estimate_time_to_collision(speed/3.6, distance_to_pedestrian, pedestrian.get_velocity())
            
            '''if distance_to_pedestrian <= braking_distance :
                target_deceleration, target_speed = self.adjust_speed_based_on_occlusion((speed+0.78)/3.6, distance_to_pedestrian - 3)
                brake = round(abs(target_deceleration)/8, 2)
                brake = 0.15
                print(brake)
                decision = {"decision": "BRAKING", "Explanation": "PEDESTRIAN OCCLUDED", "brake": brake, "throttle": 0.0, "gear": 0, "steer": 0.0, "manual_gear": 1, "auto_pilot": False ,"speed": speed }
                ped_cross = 2
                self.ego_vehicle.set_autopilot(False, self.tm.get_port())
            elif speed < (target_speed + 0.78): 
                decision = {"decision": "ACCELERATE", "Explanation": "FREE MOVEMENT", "brake": 0.0, "throttle": min(1.0, ((target_speed + 0.78)/3.6 - speed/3.6) * 2.0),
                                             "steer": 0.0, "auto_pilot": True, "gear": None,"speed": speed }
            else:
                decision = {"decision": "BRAKING", "Explanation": "REACH SPEED LIMIT", "brake": min(1.0, ((target_speed + 0.78)/3.6 - speed/3.6) * 2.0), 
                                           "throttle": 0.0,
                                             "steer": 0.0, "auto_pilot": True , "gear": None,"speed": speed }'''
            
            self.car_decisions.append(decision)
                
        elif ped_cross == 2 :
            decision = self.car_decisions[len(self.car_decisions)-1]
            if distance_to_pedestrian <= self.settings[self.scenario]["DIS_TO_EGO_REACTION"]:
                decision['brake'] = 0.64
            
            decision['speed'] = speed
            if speed < 2.0:
                decision['brake'] = 0.05
                #control.hand_brake = True
            self.car_decisions.append(decision)
        elif ped_cross == 0: #or ped_cross == 3:
            if speed < (target_speed + 0.78): 
                decision = {"decision": "ACCELERATE", "Explanation": "FREE MOVEMENT", "brake": 0.0, "throttle": min(1.0, ((target_speed + 0.78)/3.6 - speed/3.6) * 2.0),
                                             "steer": 0.0, "auto_pilot": False, "gear": None,"speed": speed }
                self.car_decisions.append(decision)
            else:
                decision = {"decision": "BRAKING", "Explanation": "REACH SPEED LIMIT", "brake": min(1.0, ((target_speed + 0.78)/3.6 - speed/3.6) * 2.0), 
                                           "throttle": 0.0,
                                             "steer": 0.0, "auto_pilot": False , "gear": None,"speed": speed }
                self.car_decisions.append(decision)
        else:
            if speed < (target_speed + 0.78): 
                decision = {"decision": "ACCELERATE", "Explanation": "FREE MOVEMENT", "brake": 0.0, "throttle": min(1.0, ((target_speed + 0.78)/3.6 - speed/3.6) * 2.0),
                                             "steer": 0.0, "auto_pilot": False, "gear": None,"speed": speed }
                self.car_decisions.append(decision)
            else:
                decision = {"decision": "STOP", "Explanation": "REACH SPEED LIMIT", "brake": 0.0, 
                                           "throttle": 0.0,
                                             "steer": 0.0, "auto_pilot": False , "gear": None,"speed": speed }
            self.car_decisions.append(decision)
        
        
        
        
        control.throttle = decision['throttle']
        control.brake = decision['brake']
        control.steer = decision['steer']
        if decision['gear'] is not None:
            control.gear = decision['gear']
            control.manual_gear_shift = decision["manual_gear"]


        #print("S:"+str(speed)+" T:"+str(target_speed)+" PRED:"+str(occlusion_prob))
        print(decision)
        # Adjust speed if occlusion is detected
        '''if occlusion_prob >= self.settings["ENHANCED_VEH"]["OCCLUSION_SENSITIVY"]:
            target_speed = self.adjust_speed_based_on_occlusion(speed, distance_to_pedestrian)
            #CALCULATE THE DESIRED SPEED IF I WANT THAT THE CAR STOP IN A DESIRED DISTANCE
        else:
            target_speed = ((self.settings[self.scenario]["EGO_SPEED"] + 0.78)/3.6 - speed/3.6) * 2.0

        print(str(target_speed))
        # Actions to take during each simulation step
        control = carla.VehicleControl()

        if distance_to_pedestrian > safe_stopping_distance:

            if speed < (self.settings[self.scenario]["EGO_SPEED"] + 0.78) :

                control.throttle = max(0.2, min(1.0, target_speed))
                control.brake = 0.0
                control.steer = 0.0
            
            else:
                # Apply braking if too close to pedestrian
                control.brake = min(1.0, (safe_stopping_distance - braking_distance) / 10)
                control.throttle = 0.0
                print("🚨 Braking to avoid pedestrian!")
        else:
            control.throttle = 0.0
            control.brake = 0.64
        '''

        return control, ped_cross, ttc_anticipation


# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, hud, settings):
        """Constructor method"""
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.ego_vehicle = None
        self.occ_vehicle = None
        self.pedestrian = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        scenario = settings['SCENARIO']
        self.restart(settings, scenario)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        

    def restart(self, settings, scenario):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0
        bp_lib = self.world.get_blueprint_library() 

        blueprint = bp_lib.find('vehicle.bmw.grandtourer')
        blueprint.set_attribute('role_name', 'ego')
        blueprint.set_attribute('color', '0,0,0')

        # Spawn the ego_vehicle.
        if self.ego_vehicle is not None:
            spawn_point_ego_veh = self.ego_vehicle.get_transform()
            spawn_point_ego_veh.location.z += 2.0
            spawn_point_ego_veh.rotation.roll = 0.0
            spawn_point_ego_veh.rotation.pitch = 0.0
            self.destroy()
            self.ego_vehicle = self.world.try_spawn_actor(blueprint, spawn_point_ego_veh)
            self.modify_vehicle_physics(self.ego_vehicle)


        # EGO VEHICLE DEFINITION
        spawn_point_ego_veh = carla.Transform(carla.Location(x=settings[scenario]['EGO_X'], y=settings[scenario]['EGO_Y'], z=settings[scenario]['EGO_Z']), carla.Rotation(yaw=settings[scenario]['EGO_YAW']))
        self.ego_vehicle = self.world.try_spawn_actor(blueprint, spawn_point_ego_veh)
        #self.ego_vehicle.set_autopilot(True)            
        self.modify_vehicle_physics(self.ego_vehicle)

        # OCCLUDED VEHICLE DEFINITION
        spawn_point_occ_veh = carla.Transform(carla.Location(x=settings[scenario]['OCC_X'], y=settings[scenario]['OCC_Y'], z=settings[scenario]['OCC_Z']), carla.Rotation(yaw=settings[scenario]['OCC_YAW']))
        occ_veh = bp_lib.find(settings[scenario]['OCC_VEH'])
        occ_veh.set_attribute('role_name', 'car2')
        self.occ_vehicle = self.world.try_spawn_actor(occ_veh, spawn_point_occ_veh)

        # PEDESTRIAN DEFINITION
        walker = bp_lib.find(settings[scenario]['WALKER'])
        walker.set_attribute('is_invincible', 'false')
        spawn_point_ped = carla.Transform(carla.Location(x=settings[scenario]['WALKER_X'], y=settings[scenario]['WALKER_Y'], z=settings[scenario]['WALKER_Z']), carla.Rotation(yaw=settings[scenario]['WALKER_YAW'])) 
        self.pedestrian = self.world.try_spawn_actor(walker, spawn_point_ped)

        #self.world.wait_for_tick()
        self.world.tick()

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.ego_vehicle, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.ego_vehicle, self.hud)
        self.gnss_sensor = GnssSensor(self.ego_vehicle)
        self.imu_sensor = ImuSensor(self.ego_vehicle)
        self.camera_manager = CameraManager(self.ego_vehicle, self.hud)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.ego_vehicle)
        self.hud.notification(actor_type)

       
    def rgb_callback(self,image, data_dict):
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)) #Reshaping with alpha channel
        #img[:,:,3] = 255 #Setting the alpha to 255 
        data_dict['rgb_image'] = img
        i3 = img[:, :, :3]
        i3 = i3[:, :, ::-1]  # Convert from BGR to RGB

        surface = pygame.surfarray.make_surface(i3.swapaxes(0, 1))

        current_time = datetime.datetime.now()
        id = str (current_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])

        captured_front_images.append((id, surface))

        return surface
        #surface = pygame.transform.scale(surface, (target_width, target_height))



    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.ego_vehicle.get_world().set_weather(preset[0])

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.ego_vehicle]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.ego_vehicle.get_transform()
        vel = world.ego_vehicle.get_velocity()
        control = world.ego_vehicle.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.ego_vehicle, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.ego_vehicle.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)


class WalkerController(object):

    def __init__(self, parent_actor, hud):
        self._parent = parent_actor
        self.hud = hud
        self.world = self._parent.get_world()
        blueprint = self.world.get_blueprint_library().find('controller.ai.walker')
        self.sensor = self.world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)

    
    def start(self):
        self.start()
        self.go_to_location(self.world.get_random_location_from_navigation())
        self.set_max_speed(1.4)
        


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))
        self.collision = False
        self.intensity = 0

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)
        self.collision = True
        self.intensity = intensity

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================

class ImuSensor(object):

    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=0, y=0, z=2)),
                                        attach_to=self._parent)
        acceleration_log = []
        self.jerk_log =  []
        self.sensor.listen(lambda event: ImuSensor.imu_callback(event, acceleration_log, self.jerk_log))

    def get_jerk(self):
        avg_jerk1, peak_jerk1 = np.mean(np.abs(self.jerk_log)), np.max(np.abs(self.jerk_log))
        return avg_jerk1, peak_jerk1
    
    def imu_callback(sensor_data, acceleration_log, jerk_log):

        # Get acceleration values
        accel_x = sensor_data.accelerometer.x
        accel_y = sensor_data.accelerometer.y
        accel_z = sensor_data.accelerometer.z

        # Compute magnitude of acceleration
        acceleration = np.linalg.norm([accel_x, accel_y, accel_z])

        # Store acceleration and timestamp
        acceleration_log.append((sensor_data.timestamp, acceleration))

        # Compute jerk if we have at least two acceleration samples
        if len(acceleration_log) > 1:
            t1, a1 = acceleration_log[-2]
            t2, a2 = acceleration_log[-1]
            jerk = (a2 - a1) / (t2 - t1) if (t2 - t1) > 0 else 0
            jerk_log.append(jerk)

class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z)), attachment.Rigid),
            (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), attachment.SpringArmGhost),
            (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))
    
    
    
  

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================


captured_front_images = []
captured_images_id = 0

def process_image(image, target_width, target_height, flip_horizontal=False, save_image=False):

    """Processes an image from CARLA to a Pygame surface and stores it in memory."""
    i = np.array(image.raw_data)
    i2 = i.reshape((image.height, image.width, 4))
    i3 = i2[:, :, :3]
    i3 = i3[:, :, ::-1]  # Convert from BGR to RGB


    surface = pygame.surfarray.make_surface(i3.swapaxes(0, 1))
    surface = pygame.transform.scale(surface, (target_width, target_height))
    if flip_horizontal:
        surface = pygame.transform.flip(surface, True, False)

    # Append the surface or its data to the list
    if save_image:
        captured_front_images.append(surface)

    return surface

def save_images_to_disk(output_dir, images):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for relative_frame_number, image in images:
        image_path = os.path.join(output_dir, f"frame_{relative_frame_number}.png")
        #image = pygame.transform.scale(image, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        pygame.image.save(image, image_path)


def calculate_jerk(accel, acceleration_log, jerk_log):
    acceleration = np.linalg.norm([accel.x, accel.y, accel.z])
    acceleration_log.append((time.time(), acceleration))
    if len(acceleration_log) > 1:
        t1, a1 = acceleration_log[-2]
        t2, a2 = acceleration_log[-1]
        jerk = (a2 - a1) / (t2 - t1) if (t2 - t1) > 0 else 0
        jerk_log.append(jerk)
    return acceleration_log, jerk_log

def game_loop(settings, conf_model):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """
    
    pygame.init()
    pygame.font.init()
    world = None

    try:
        #random.seed(args.seed)

        client = carla.Client(settings['HOST'], settings['PORT'])
        client.set_timeout(60.0)
        scenario = settings['SCENARIO']
        client.load_world(settings[scenario]['TOWN'])

        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()

        world_settings = sim_world.get_settings()
        world_settings.synchronous_mode = True
        world_settings.fixed_delta_seconds = 0.05
        sim_world.apply_settings(world_settings)
        traffic_manager.set_synchronous_mode(True)


        display = pygame.display.set_mode(
            (settings['WIDTH'], settings['HEIGHT']),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(settings['WIDTH'], settings['HEIGHT'])
        world = World(client.get_world(), hud, settings)
        controller = KeyboardControl(world)


        agent = CustomAgent(world.ego_vehicle, settings, conf_model, traffic_manager)


        camera_bp = sim_world.get_blueprint_library().find('sensor.camera.rgb') 
        camera_bp.set_attribute('image_size_x', '1920')
        camera_bp.set_attribute('image_size_y', '1080')
        camera_bp.set_attribute('fov', '110') # Field of view in degrees

        camera_init_trans = carla.Transform(carla.Location(x =-0.1,z=1.7)) 
        camera = sim_world.spawn_actor(camera_bp, camera_init_trans, attach_to=world.ego_vehicle)

        image_w = int(camera.attributes['image_size_x'])
        image_h = int(camera.attributes['image_size_y'])

        sensor_data = {'rgb_image': np.zeros((image_h, image_w, 4))}

        camera.listen(lambda image: world.rgb_callback(image, sensor_data))

        start_reaction_time = None
        reaction_time = float('inf')
        min_ttc = float('inf')
        experiment_results = {'reaction-time': None, 'ttc': None, 'collision': None, 'collision_intensity':0, 'avg-jerk': None,
                              'peak-jerk': None, 'ttc_anticipation': 0}
        acceleration_log = []
        jerk_log=[]
        ped_cross = 0
        
        clock = pygame.time.Clock()

        while True:
            clock.tick(20)
            world.world.tick()
            #world.world.wait_for_tick()
            if controller.parse_events():
                return

            world.tick(clock)
            world.render(display)
            pygame.display.flip()

            
            v_2 = world.occ_vehicle.get_velocity()
            occ_veh_speed = 3.6*math.sqrt(v_2.x**2 + v_2.y**2 +v_2.z**2)

            if settings[scenario]['AXIS'] == 'Y':
                distance_to_cross_occ_veh = world.occ_vehicle.get_location().y - settings[scenario]['OCC_DIS_CROSS']
                value_control_oclu = distance_to_cross_occ_veh > settings[scenario]['DIS_TO_CONTROL_OCC']    
                value_control_ped = distance_to_cross_occ_veh < settings[scenario]['DIS_TO_WALKER_START']
            elif settings[scenario]['AXIS'] == 'X2':
                distance_to_cross_occ_veh = world.occ_vehicle.get_location().x - settings[scenario]['OCC_DIS_CROSS']
                value_control_oclu = distance_to_cross_occ_veh > settings[scenario]['DIS_TO_CONTROL_OCC']
                value_control_ped = distance_to_cross_occ_veh < settings[scenario]['DIS_TO_WALKER_START']
            else:
                distance_to_cross_occ_veh = world.occ_vehicle.get_location().x - settings[scenario]['OCC_DIS_CROSS']
                value_control_oclu = distance_to_cross_occ_veh < settings[scenario]['DIS_TO_CONTROL_OCC']
                value_control_ped = distance_to_cross_occ_veh > settings[scenario]['DIS_TO_WALKER_START']
            

            if value_control_oclu:
                if occ_veh_speed < (settings[scenario]['OCC_SPEED'] + 0.78):
                    world.occ_vehicle.apply_control(carla.VehicleControl(throttle=min(1.0, ((settings[scenario]['OCC_SPEED'] +
                                                        0.78)/3.6 - occ_veh_speed/3.6) * 2.0), steer=0.0, brake=0.0))
                else:
                    world.occ_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0,
                                                        brake=min(1.0, (occ_veh_speed/3.6 - (settings[scenario]['OCC_SPEED'] + 0.78)/3.6) * 2.0)) )
            else:
                world.occ_vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0,
                                                    brake=settings['DECELERATION'], gear = 0, manual_gear_shift = 1))

            control, ped_cross, ttc_anticipation = agent.run_step(world.pedestrian, world.occ_vehicle, ped_cross)
            #control.manual_gear_shift = False
            world.ego_vehicle.apply_control(control)

            if ttc_anticipation is not None:
                experiment_results['ttc_anticipation'] = ttc_anticipation

            # CALCULATE THE REACTION TIME
            distance_to_pedestrian_ego_veh = world.ego_vehicle.get_location().distance(world.pedestrian.get_location())
            if distance_to_pedestrian_ego_veh <= settings['DETECTION_THRESHOLD'] and start_reaction_time is None:
                start_reaction_time = time.time()
            
            if start_reaction_time and world.ego_vehicle.get_control().brake > 0.10 and experiment_results['reaction-time'] is None:
                reaction_time = time.time() - start_reaction_time
                experiment_results['reaction-time'] = reaction_time

            # CALCULATE THE TIME TO COLLISION
            ego_velocity = world.ego_vehicle.get_velocity()
            pedestrian_velocity = world.pedestrian.get_velocity()
            #ego_speed = np.linalg.norm([ego_velocity.x, ego_velocity.y, ego_velocity.z])
            #pedestrian_speed = np.linalg.norm([pedestrian_velocity.x, pedestrian_velocity.y, pedestrian_velocity.z])
            ego_speed = math.sqrt(ego_velocity.x**2 + ego_velocity.y**2 +ego_velocity.z**2)
            pedestrian_speed = math.sqrt(pedestrian_velocity.x**2 + pedestrian_velocity.y**2 +pedestrian_velocity.z**2)
            relative_speed = ego_speed - pedestrian_speed
            if relative_speed > 0:
                ttc = distance_to_pedestrian_ego_veh / relative_speed
            else:
                ttc = float('inf')

            if ttc < min_ttc:
                min_ttc = ttc
                experiment_results['ttc'] = min_ttc

            if  world.collision_sensor.collision  and experiment_results['collision'] is None:
                experiment_results['collision'] = True
                experiment_results['collision_intensity'] = world.collision_sensor.intensity

            if ego_speed *3.6 > 5 and world.collision_sensor.collision == False:
                if world.ego_vehicle.get_location().x <= settings[scenario]['WALKER_X'] :
                    acceleration_log, jerk_log = calculate_jerk(world.ego_vehicle.get_acceleration(), acceleration_log, jerk_log)


            #print("D:"+ str(distance_to_cross_occ_veh)+" - V:"+str(occ_veh_speed))
            
            if occ_veh_speed < settings[scenario]['VEL_TO_WALKER_START'] and value_control_ped:
                
                walker_control = carla.WalkerControl()
                walker_direction = settings[scenario]['WALKER_DIRECTION']
                walker_control.direction = carla.Vector3D(walker_direction[0], walker_direction[1], walker_direction[2]) 
                walker_control.speed = settings[scenario]['WALKER_SPEED']
                world.pedestrian.apply_control(walker_control)
                if world.pedestrian.get_location().distance(carla.Location(x= settings[scenario]['WALKER_X_2'], y= settings[scenario]['WALKER_Y_2'], z= settings[scenario]['WALKER_Z_2'])) < 0.30:
                    ped_cross = 3

    

    finally:

        if world is not None:
            sett = world.world.get_settings()
            sett.synchronous_mode = False
            sett.fixed_delta_seconds = None
            world.world.apply_settings(sett)
            traffic_manager.set_synchronous_mode(True)
            #experiment_results['avg-jerk'] =np.mean(np.abs(jerk_log))
            #experiment_results['peak-jerk'] = np.max(np.abs(jerk_log))
            print(experiment_results)

            if settings['SAVE_IMAGES']:
                save_images_to_disk("video", captured_front_images)

            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    with open('scenarios.yaml') as f:
        settings = yaml.load(f, Loader=SafeLoader)


    with open('conf_model.yaml') as f:
        conf_model = yaml.load(f, Loader=SafeLoader)
    
    log_level = logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    print(__doc__)

    try:
        game_loop(settings, conf_model)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()

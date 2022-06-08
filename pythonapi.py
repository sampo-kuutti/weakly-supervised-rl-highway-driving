# Author:Author: Sampo Kuutti (s.j.kuutti@surrey.ac.uk)
# Organisation: University of Surrey
#
# pythonapi.py imports the api.dll which provides functionality for communicating with IPG Carmaker
# the file also defines all the necessary functions from the c api.dll for reading relevant simulation data
# as well as controlling the host vehicle gas and brake pedals

# Note, if you don't have access to the shared safeav directory, you will need to change the API dll path to where
# you have the ipg api saved.
# =====================================================================================================================

import ctypes

# change path to dll if needed
_mod=ctypes.cdll.LoadLibrary("S:/Research/safeav/Supervised Learning/Sample_API/API.dll")
# load dll

ApoClnt_PollAndSleep = _mod.ApoClnt_PollAndSleep
# ApoClnt_PollAndSleep will be called from within the ApoClnt code whenever it needs to wait
# for something without blocking the rest of the client application.

api_setup = _mod.api_setup
# sets up the connection between the api and the carmaker client

api_terminate = _mod.api_terminate
# terminates the connection between API and IPG CarMaker

sim_start = _mod.sim_start
# starts the simulation with currently loaded simulation scenario

sim_waitready = _mod.sim_waitready
# waits until simulation is running

sim_stop = _mod.sim_stop
# stops the simulation

sim_isrunning = _mod.sim_isrunning
sim_isrunning.restype = ctypes.c_int
# checks if simulation is running

sim_loadrun = _mod.sim_loadrun
sim_loadrun.argtype = ctypes.c_int
# loads a testrun from a predefined list of testruns

sim_loadrun2 = _mod.sim_loadrun2
sim_loadrun2.argtype = ctypes.c_int
# loads a testrun from a predefined list of testruns

subscribe_quants = _mod.subscribe_quants
#subscribes to selected quantities

get_radardist = _mod.get_radardist
get_radardist.restype = ctypes.c_double
# returns measured radar distance to nearest object ahead of the host vehicle

get_radarvel = _mod.get_radarvel
get_radarvel.restype = ctypes.c_double
# returns measured radar relativ velocity to nearest object ahead of the host vehicle

get_hostvel = _mod.get_hostvel
get_hostvel.restype = ctypes.c_double
# returns host vehicle velocity

get_longacc = _mod.get_longacc
get_longacc.restype = ctypes.c_double
# returns host vehicle longitudinal acceleration

get_time = _mod.get_time
get_time.restype = ctypes.c_double
# returns current time of the simulation run

get_gas = _mod.get_gas
get_gas.restype = ctypes.c_double
# returns current gas pedal value

get_brake = _mod.get_brake
get_brake.restype = ctypes.c_double
# returns current brake pedal value

get_hostpos = _mod.get_hostpos
get_hostpos.restype = ctypes.c_double
# returns position of the host vehicle

set_gas = _mod.set_gas
set_gas.argtype = ctypes.c_double
# sets the host vehicle gas pedal to the desired value, input is a double which should be between 0 and 1

set_brake = _mod.set_brake
set_brake.argtype = ctypes.c_double
# similar to set_gas, set_brake sets the brake pedal value, with an input between 0 and 1

set_radardist = _mod.set_radardist
set_radardist.argtype = ctypes.c_double
# set radar distance equal to input value

set_radarvel = _mod.set_radarvel
set_radarvel.argtype = ctypes.c_double
# set radar velocity equal to input value

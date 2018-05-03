#!/usr/bin/env python
'''
Created on Apr 4, 2012

@author: lanquarden
'''
import sys
import argparse
import socket
import time

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

# Configure the argument parser
parser = argparse.ArgumentParser(description = 'Python client to connect to the TORCS SCRC server.')

parser.add_argument('--host', action='store', dest='host_ip', default='localhost', help='Host IP address (default: localhost)')
parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001, help='Host port number (default: 3001)')
parser.add_argument('--id', action='store', dest='id', default='SCR', help='Bot ID (default: SCR)')
parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=100, help='Maximum number of learning episodes (default: 100)')
parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0, help='Maximum number of steps (default: 0)')
parser.add_argument('--track', action='store', dest='track', default=None, help='Name of the track')
parser.add_argument('--stage', action='store', dest='stage', type=int, default=3, help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')

antarg = parser.add_argument_group('Agent')
antarg.add_argument("--algorithm", choices=['network', 'hardcoded'], default="network", help="Which algorithm to use for driving the car.")
antarg.add_argument("--device", choices=['wheel', 'keyboard'], default="wheel", help="Whether to driving wheel or keyboard to control the car.")
antarg.add_argument("--mode", choices=['override', 'ff'], default="override", help="Whether to use manual control override or force feedback to showcase moves.")

antarg.add_argument("--joystick_nr", type=int, default=0, help="Joystick number in case of many.")
antarg.add_argument("--autocenter", type=int, default=20, help="Autocenter for force feedback wheels.")
antarg.add_argument("--gain", type=int, default=100, help="Gain for force feedback wheels.")
antarg.add_argument("--min_force", type=float, default=0.005, help="Only apply force if stronger than this.")
antarg.add_argument("--min_level", type=int, default=0x1000, help="Minimal force level to apply.")
antarg.add_argument("--max_level", type=int, default=0x4000, help="Maximal force level to apply.")
antarg.add_argument("--maxwheelsteps", type=int, default=50, help="How many steps wheel control persists after moving the wheel.")
antarg.add_argument("--max_speed", type=int, default=0, help="Maximum speed during driving.")
antarg.add_argument("--max_terminal_steps", type=int, default=250, help="How many steps manual control persists after going out of track.")

antarg.add_argument("--enable_training", type=str2bool, default=True, help="Enable training, by default True.")
antarg.add_argument("--enable_exploration", type=str2bool, default=True, help="Enable exploration, by default True.")
antarg.add_argument("--pretrained_network", default="models/zura_test_55.pkl", help="Pretrained network to load when appropriate button is pressed.")

antarg.add_argument("--exploration_rate_start", type=float, default=1.0, help="Exploration rate at the beginning of decay.")
antarg.add_argument("--exploration_rate_end", type=float, default=0.0, help="Exploration rate at the end of decay.")
antarg.add_argument("--exploration_decay_steps", type=int, default=10000, help="How many steps to decay the exploration rate.")
antarg.add_argument("--skip", type=int, default=0, help="Use the same action for this number of consecutive states.")

antarg.add_argument("--show_sensors", type=str2bool, default=False, help="Show sensors.")
antarg.add_argument("--update_sensors_interval", type=int, default=1, help="Update sensor values after every x steps.")
antarg.add_argument("--show_qvalues", type=str2bool, default=False, help="Show Q-values.")
antarg.add_argument("--update_qvalues_interval", type=int, default=1, help="Update Q-values after every x steps.")

antarg.add_argument("--save_csv", help="Save results in CSV file.")

memarg = parser.add_argument_group('Replay memory')
memarg.add_argument("--replay_size", type=int, default=1000000, help="Maximum size of replay memory.")
memarg.add_argument("--load_replay", help="Load replay memory from this file.")
memarg.add_argument("--save_replay", help="Save replay memory to this file at the end of training.")

netarg = parser.add_argument_group('Deep Q-learning network')
netarg.add_argument("--learning_rate", type=float, default=0.00025, help="Learning rate.")
netarg.add_argument("--discount_rate", type=float, default=0.99, help="Discount rate for future rewards.")
netarg.add_argument("--batch_size", type=int, default=32, help="Batch size for neural network.")
netarg.add_argument('--optimizer', choices=['rmsprop', 'adam', 'adadelta'], default='adadelta', help='Network optimization algorithm.')
netarg.add_argument("--decay_rate", type=float, default=0.95, help="Decay rate for RMSProp and Adadelta algorithms.")
netarg.add_argument("--clip_error", type=float, default=0, help="Clip error term in update between this number and its negative.")

netarg.add_argument("--hidden_nodes", type=int, default=50, help="Number of nodes in hidden layer.")
netarg.add_argument("--hidden_layers", type=int, default=1, help="Number of hidden layers.")

netarg.add_argument("--target_steps", type=int, default=10000, help="Copy main network to target network after this many steps.")
netarg.add_argument("--load_weights", help="Load network from file.")
netarg.add_argument("--save_weights_prefix", default="test", help="After each epoch save network to given file. Epoch and extension will be appended.")
netarg.add_argument("--save_interval", type=int, default=100, help="Save weights after this many episodes.")

neonarg = parser.add_argument_group('Neon')
neonarg.add_argument('--backend', choices=['cpu', 'gpu'], default='gpu', help='backend type')
neonarg.add_argument('--device_id', type=int, default=0, help='gpu device id (only used with GPU backend)')
neonarg.add_argument('--datatype', choices=['float16', 'float32', 'float64'], default='float32', help='default floating point precision for backend [f64 for cpu only]')
neonarg.add_argument('--stochastic_round', const=True, type=int, nargs='?', default=False, help='use stochastic rounding [will round to BITS number of bits if specified]')

comarg = parser.add_argument_group('Common')
comarg.add_argument("--random_seed", type=int, help="Random seed for repeatable experiments.")
comarg.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Log level.")
comarg.add_argument("--verbose", type=str2bool, default=False, help="Enable debugging information.")


# name of the driver to use 
parser.add_argument('--driver', choices=['orig', 'key', 'wheel', 'ff', 'random', 'dqn', 'linear', 'demo', 'ac', 'alinear', 'record', 'player'], default='alinear')
arguments = parser.parse_args()

# Print summary
print 'Connecting to server host ip:', arguments.host_ip, '@ port:', arguments.host_port
print 'Bot ID:', arguments.id
print 'Maximum episodes:', arguments.max_episodes
print 'Maximum steps:', arguments.max_steps
print 'Track:', arguments.track
print 'Stage:', arguments.stage
print 'Driver:', arguments.driver
print '*********************************************'

if arguments.driver == 'orig':
    from origdriver import Driver
    driver = Driver(arguments.stage)
elif arguments.driver == 'key':
    from keydriver import Driver
    driver = Driver(arguments.stage)
elif arguments.driver == 'wheel':
    from wheeldriver import Driver
    driver = Driver(arguments.stage)
elif arguments.driver == 'ff':
    from ffdriver import Driver
    driver = Driver(arguments.stage)
elif arguments.driver == 'random':
    from randomdriver import Driver
    driver = Driver(arguments.stage)
elif arguments.driver == 'dqn':
    from dqndriver import Driver
    driver = Driver(arguments)
elif arguments.driver == 'linear':
    from lineardriver import Driver
    driver = Driver(arguments)
elif arguments.driver == 'alinear':
    from alineardriver import Driver
    driver = Driver(arguments)
elif arguments.driver == 'demo':
    from demodriver import Driver
    driver = Driver(arguments)
elif arguments.driver == 'ac':
    from acdriver import Driver
    driver = Driver(arguments)
elif arguments.driver == 'record':
    from recorddriver import Driver
    driver = Driver(arguments)
elif arguments.driver == 'player':
    from playerdriver import Driver
    driver = Driver(arguments)
else:
    assert False, "Unknown driver"

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except socket.error, msg:
    print 'Could not make a socket.'
    sys.exit(-1)

# one second timeout
sock.settimeout(1.0)

shutdownClient = False
curEpisode = 0

verbose = arguments.verbose

while True:
    while True:
        if verbose:
          print 'Sending id to server: ', arguments.id
        buf = arguments.id + driver.init()
        if verbose:
          print 'Sending init string to server:', buf
        
        try:
            sock.sendto(buf, (arguments.host_ip, arguments.host_port))
        except socket.error, msg:
            print "Failed to send data...Exiting..."
            sys.exit(-1)

            
        try:
            buf, addr = sock.recvfrom(1000)
        except socket.error, msg:
            print "didn't get response from server... %s" % msg
    
        if buf.find('***identified***') >= 0:
            if verbose:
                print 'Received: ', buf
            break

    currentStep = 0
    
    while True:
        # wait for an answer from server
        buf = None
        try:
            buf, addr = sock.recvfrom(1000)
        except socket.error, msg:
            print "didn't get response from server... %s" % msg
        
        if verbose:
            print 'Received: ', buf
        
        if buf != None and buf.find('***shutdown***') >= 0:
            driver.onShutDown()
            shutdownClient = True
            if verbose:
              print 'Client Shutdown'
            break
        
        if buf != None and buf.find('***restart***') >= 0:
            driver.onRestart()
            if verbose:
              print 'Client Restart'
            time.sleep(1)
            break
        
        currentStep += 1
        if currentStep != arguments.max_steps:
            if buf != None:
                buf = driver.drive(buf)
        else:
            buf = '(meta 1)'
        
        if verbose:
            print 'Sending: ', buf
        
        if buf != None:
            try:
                sock.sendto(buf, (arguments.host_ip, arguments.host_port))
            except socket.error, msg:
                print "Failed to send data...Exiting..."
                sys.exit(-1)
    
    curEpisode += 1
    
    if curEpisode == arguments.max_episodes:
        driver.onShutDown()
        shutdownClient = True
        
    if curEpisode % 2 == 0:
        driver.reset()
    else:
        driver.test_mode()

sock.close()

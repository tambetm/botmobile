#!/usr/bin/env python
'''
@authors: RDTm, tambetm
'''

import logging
logging.basicConfig(format='%(asctime)s %(message)s')

import sys
import argparse
import socket
import driver_speed
from replay_memory_speed import ReplayMemory
from deepqnetwork_speed import DeepQNetwork
import random

if __name__ == '__main__':
    pass

# Configure the argument parser
parser = argparse.ArgumentParser(description='Python client to connect to the TORCS SCRC server.')
mainarg = parser.add_argument_group('Main loop')

mainarg.add_argument('--host', action='store', dest='host_ip', default='localhost', help='Host IP address (default: localhost)')
mainarg.add_argument('--port', action='store', type=int, dest='host_port', default=3001, help='Host port number (default: 3001)')
mainarg.add_argument('--id', action='store', dest='id', default='SCR', help='Bot ID (default: SCR)')
mainarg.add_argument('--max_episodes', action='store', dest='max_episodes', type=int, default=1, help='Maximum number of learning episodes (default: 1)')
mainarg.add_argument('--max_steps', action='store', dest='max_steps', type=int, default=0, help='Maximum number of steps (default: 0)')
mainarg.add_argument('--track', action='store', dest='track', default=None, help='Name of the track')
mainarg.add_argument('--stage', action='store', dest='stage', type=int, default=3, help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')

mainarg.add_argument("--load_weights", help="Load network from file.")
mainarg.add_argument("--save_weights_prefix", help="Save network to given file. Epoch and extension will be appended.")
mainarg.add_argument("--csv_file", help="Write training progress to this file.")

# Things we might need eventually
#  parser.add_argument("--frame_skip", type=int, default=4, help="How many times to repeat each chosen action.")

# Arguments related to keeping memory
memarg = parser.add_argument_group('Replay memory')
memarg.add_argument("--replay_size", type=int, default=1000000, help="Maximum size of replay memory.")
#memarg.add_argument("--history_length", type=int, default=4, help="How many screen frames form a state.")

#  Arguments related to the DQN underlying the BotMobile
netarg = parser.add_argument_group('Deep Q-learning network')
netarg.add_argument("--learning_rate", type=float, default=0.00025, help="Learning rate.")
netarg.add_argument("--discount_rate", type=float, default=0.99, help="Discount rate for future rewards.")
netarg.add_argument("--batch_size", type=int, default=32, help="Batch size for neural network.")
netarg.add_argument('--optimizer', choices=['rmsprop', 'adam', 'adadelta'], default='rmsprop', help='Network optimization algorithm.')
netarg.add_argument("--decay_rate", type=float, default=0.95, help="Decay rate for RMSProp and Adadelta algorithms.")
netarg.add_argument("--clip_error", type=float, default=1, help="Clip error term in update between this number and its negative. Default == 1. If 0, then there is no error clipping. ")
netarg.add_argument("--target_steps", type=int, default=0, help="Copy main network to target network after this many steps. Default == 0. If 0, then there is only one network")

# Arguments related to the Neural Network implementation in Neon
neonarg = parser.add_argument_group('Neon')
neonarg.add_argument('--backend', choices=['cpu', 'gpu'], default='cpu', help='backend type')
neonarg.add_argument('--device_id', type=int, default=0, help='gpu device id (only used with GPU backend)')
neonarg.add_argument('--datatype', choices=['float16', 'float32', 'float64'], default='float32', help='default floating point precision for backend [f64 for cpu only]')
neonarg.add_argument('--stochastic_round', const=True, type=int, nargs='?', default=False, help='use stochastic rounding [will round to BITS number of bits if specified]')

# Arguments related to the BotMobile
botarg = parser.add_argument_group('Agent')
botarg.add_argument("--exploration_rate_start", type=float, default=0.5, help="Exploration rate at the beginning of decay.")
botarg.add_argument("--exploration_rate_end", type=float, default=0.05, help="Exploration rate at the end of decay.")
botarg.add_argument("--exploration_decay_steps", type=float, default=10000, help="After how many steps to decay the exploration rate.")
botarg.add_argument("--exploration_rate_test", type=float, default=0.05, help="Exploration rate used during testing.")
botarg.add_argument("--train_frequency", type=int, default=4, help="Perform training after this many game steps.")
botarg.add_argument("--train_repeat", type=int, default=1, help="Number of times to sample minibatch during training.")

commonarg = parser.add_argument_group('Common')
commonarg.add_argument("--random_seed", type=int, help="Random seed for repeatable experiments.")
commonarg.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Log level.")
args = parser.parse_args()

logger = logging.getLogger()
logger.setLevel(args.log_level)

if args.random_seed:
    random.seed(args.random_seed)

mem = ReplayMemory(args.replay_size, args)
net = DeepQNetwork(5, args)
agent = driver_speed.Driver(mem, net, args)

if args.load_weights:
  logger.info("Loading weights from %s" % args.load_weights)
  net.load_weights(args.load_weights)

# Print summary
print 'Connecting to server host ip:', args.host_ip, '@ port:', args.host_port
print 'Bot ID:', args.id
print 'Maximum episodes:', args.max_episodes
print 'Maximum steps:', args.max_steps
print 'Track:', args.track
print 'Stage:', args.stage
print '*********************************************'

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except socket.error, msg:
    print 'Could not make a socket.'
    sys.exit(-1)

# one second timeout
sock.settimeout(1.0)
shutdownClient = False
curEpisode = 0
verbose = False


while not shutdownClient:

    # This is not the main loop, this loop just initializes communication
    while True:
        print 'Sending id to server: ', args.id
        buf = args.id + agent.init()  # agent sends the angles of distance sensors
        print 'Sending init string to server:', buf
        
        try:
            sock.sendto(buf, (args.host_ip, args.host_port))
        except socket.error, msg:
            print "Failed to send data...Exiting..."
            sys.exit(-1)
            
        try:
            buf, addr = sock.recvfrom(1000)
        except socket.error, msg:
            print "didn't get response from server..."
    
        if buf.find('***identified***') >= 0:
            print 'Received: ', buf
            break

    # This is the main loop that sends actions
    currentStep = 0
    while True:
        # wait for an answer from server
        buf = None
        try:
            buf, addr = sock.recvfrom(1000)
        except socket.error, msg:
            print "didn't get response from server..."
        
        if verbose:
            print 'Received: ', buf
        
        if buf != None and buf.find('***shutdown***') >= 0:
            agent.onShutDown()
            shutdownClient = True
            print 'Client Shutdown'
            break
        
        if buf != None and buf.find('***restart***') >= 0:
            agent.onRestart()
            print 'Client Restart'
            break
        
        currentStep += 1
        if currentStep != args.max_steps:
            if buf != None:
                buf = agent.drive(buf)
            else:
                print "We were supposed to receive information to choose and action, but buffer was empty"
        # if currents steps equals max, restart the race (meta 1)
        else:
            buf = '(meta 1)'
        
        if verbose:
            print 'Sending: ', buf
        
        if buf != None:
            try:
                sock.sendto(buf, (args.host_ip, args.host_port))
            except socket.error, msg:
                print "Failed to send data...Exiting..."
                sys.exit(-1)

    # When a game has ended we would like to save the current model for future generations to see
    if args.save_weights_prefix:
        filename = args.save_weights_prefix + "_%d.pkl" % (curEpisode + 1)
        logger.info("Saving weights to %s" % filename)
        net.save_weights(filename)

    curEpisode += 1
    
    if curEpisode == args.max_episodes:
        shutdownClient = True
        

sock.close()

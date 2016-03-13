#!/usr/bin/env python
'''
Created on Apr 4, 2012

@author: lanquarden
'''
import sys
import argparse
import socket

#if __name__ == '__main__':
#    pass

# Configure the argument parser
parser = argparse.ArgumentParser(description = 'Python client to connect to the TORCS SCRC server.')

parser.add_argument('--host', action='store', dest='host_ip', default='localhost',
                    help='Host IP address (default: localhost)')
parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001,
                    help='Host port number (default: 3001)')
parser.add_argument('--id', action='store', dest='id', default='SCR',
                    help='Bot ID (default: SCR)')
parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=100,
                    help='Maximum number of learning episodes (default: 100)')
parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0,
                    help='Maximum number of steps (default: 0)')
parser.add_argument('--track', action='store', dest='track', default=None,
                    help='Name of the track')
parser.add_argument('--stage', action='store', dest='stage', type=int, default=3,
                    help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')

antarg = parser.add_argument_group('Agent')
antarg.add_argument("--exploration_rate_start", type=float, default=1, help="Exploration rate at the beginning of decay.")
antarg.add_argument("--exploration_rate_end", type=float, default=0.1, help="Exploration rate at the end of decay.")
antarg.add_argument("--exploration_decay_steps", type=float, default=50000, help="How many steps to decay the exploration rate.")
antarg.add_argument("--show_qvalues", action="store_true", help="Show Q-values.")
antarg.add_argument("--show_sensors", action="store_true", help="Show sensors.")
antarg.add_argument("--manual_control", action="store_true", help="Allow manual control.")

memarg = parser.add_argument_group('Replay memory')
memarg.add_argument("--replay_size", type=int, default=1000000, help="Maximum size of replay memory.")

netarg = parser.add_argument_group('Deep Q-learning network')
netarg.add_argument("--learning_rate", type=float, default=0.00025, help="Learning rate.")
netarg.add_argument("--discount_rate", type=float, default=0.99, help="Discount rate for future rewards.")
netarg.add_argument("--batch_size", type=int, default=32, help="Batch size for neural network.")
netarg.add_argument('--optimizer', choices=['rmsprop', 'adam', 'adadelta'], default='rmsprop', help='Network optimization algorithm.')
netarg.add_argument("--decay_rate", type=float, default=0.95, help="Decay rate for RMSProp and Adadelta algorithms.")
netarg.add_argument("--clip_error", type=float, default=1, help="Clip error term in update between this number and its negative.")
netarg.add_argument("--target_steps", type=int, default=10000, help="Copy main network to target network after this many steps.")
netarg.add_argument("--load_weights", help="Load network from file.")
netarg.add_argument("--save_weights_prefix", default="test", help="Save network to given file. Epoch and extension will be appended.")

neonarg = parser.add_argument_group('Neon')
neonarg.add_argument('--backend', choices=['cpu', 'gpu'], default='gpu', help='backend type')
neonarg.add_argument('--device_id', type=int, default=0, help='gpu device id (only used with GPU backend)')
neonarg.add_argument('--datatype', choices=['float16', 'float32', 'float64'], default='float32', help='default floating point precision for backend [f64 for cpu only]')
neonarg.add_argument('--stochastic_round', const=True, type=int, nargs='?', default=False, help='use stochastic rounding [will round to BITS number of bits if specified]')

comarg = parser.add_argument_group('Common')
comarg.add_argument("--random_seed", type=int, help="Random seed for repeatable experiments.")
comarg.add_argument("--log_level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO", help="Log level.")

# name of the driver to use 
# driver built in, manual control by keyboard
parser.add_argument('--driver', choices=['orig', 'key', 'wheel', 'ff', 'tm', 'random'], default='tm')
arguments = parser.parse_args()

# Print summary
print 'Connecting to server host ip:', arguments.host_ip, '@ port:', arguments.host_port
print 'Bot ID:', arguments.id
print 'Maximum episodes:', arguments.max_episodes
print 'Maximum steps:', arguments.max_steps
print 'Track:', arguments.track
print 'Stage:', arguments.stage
print 'driver:', arguments.driver
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

if arguments.driver == 'key':
    import keydriver
    d = keydriver.KeyDriver(arguments.stage)
elif arguments.driver == 'wheel':
    import wheeldriver
    d = wheeldriver.WheelDriver(arguments.stage)
elif arguments.driver == 'ff':
    import ffdriver
    d = ffdriver.ForceFeedbackDriver(arguments.stage)
elif arguments.driver == 'random':
    import randdriver
    d = randdriver.RandDriver(arguments.stage)
elif arguments.driver == 'tm':
    from tmdriver import Driver
    from replay_memory import ReplayMemory
    from deepqnetwork import DeepQNetwork
    mem = ReplayMemory(arguments.replay_size, 19, arguments)
    net = DeepQNetwork(19, 26, arguments)
    d = Driver(arguments.stage, net, mem, arguments)
elif arguments.driver == 'orig':
    import origdriver
    d = origdriver.OrigDriver(arguments.stage)
else:
    assert False, "Unknown driver"

while not shutdownClient:
    while True:
        print 'Sending id to server: ', arguments.id
        buf = arguments.id + d.init()
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
            d.onShutDown()
            shutdownClient = True
            print 'Client Shutdown'
            break
        
        if buf != None and buf.find('***restart***') >= 0:
            d.onRestart()
            print 'Client Restart'
            break
        
        currentStep += 1
        if currentStep != arguments.max_steps:
            if buf != None:
                buf = d.drive(buf)
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
        shutdownClient = True
        

sock.close()

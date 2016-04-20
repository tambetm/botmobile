import argparse
import numpy as np

def parseInts(rng):
    ids = []
    for x in map(str.strip,rng.split(',')):
        if x.isdigit():
        	ids.append(int(x))
        	continue
        if x[0] == '<':
        	ids.extend(range(1,int(x[1:])+1))
        	continue
        if '-' in x:
        	xr = map(str.strip,x.split('-'))
        	ids.extend(range(int(xr[0]),int(xr[1])+1))
        	continue
        else:
            raise Exception, 'unknown range type: "%s"'%x
    return ids
        
parser = argparse.ArgumentParser()
parser.add_argument('session_file')
parser.add_argument('model_file')
parser.add_argument('--state_fields', type=parseInts, default="17-35")
parser.add_argument('--action_fields', type=parseInts, default="6,3,5")
args = parser.parse_args()

print "State fields:", args.state_fields
print "Action fields:", args.action_fields

print "Loading data from", args.session_file
data = np.load(args.session_file)
states = data['states'][:,args.state_fields]
actions = data['actions'][:,args.action_fields]
print "states:", states.shape, "actions:", actions.shape

print "Training model..."
states = np.hstack((states, np.ones((states.shape[0], 1))))
coeff = np.linalg.lstsq(states, actions)[0]
assert coeff.shape == (states.shape[1], actions.shape[1]), "states:"+str(states.shape)+" coeff:"+str(coeff.shape)

print "Saving model to", args.model_file
np.save(args.model_file, coeff)
print "Done"


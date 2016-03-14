import csv
import copy

class CSVLogger(object):
    def __init__(self, fn):
        ''' take filename for a logfile '''
        self.rows = []
        self.nrow = 0
        self.writer = None
        self.fn = fn
        self.inevery = 500
    
    def separate(self, k, arr, dct):
        # separates array to write into csv
        # for example track = [1, 2, 3], returns 
        # {'track0', 1} , {'rack1', 2} ... , adds them to dct
        for i, v in enumerate(arr):
            dct[k + str(i)] = v
            

    def to_csv_dict(self, state):
        ret = {}
        for k, val in state.items():
            if k == 'sensors' or k == 'parser':
                continue
            if type(val) == list:
                self.separate(k, val, ret)
        return ret



    def log(self, state):
        #import ipdb; ipdb.set_trace()
        
        dct = self.to_csv_dict(state.__dict__)
        dct["acc"] = state.acc
        dct["wheel"] = state.wheel
        dct["brake"] = state.brake
        if self.nrow == 0:
            with open(self.fn, 'w') as f:
                # write headers
                self.writer = csv.DictWriter(f, dct.keys())
                self.writer.writeheader()
        
        
        self.nrow += 1
        self.rows.append(dct)
        if len(self.rows) % self.inevery == 0:
            #import ipdb; ipdb.set_trace()
            with open(self.fn, 'a') as f:
                self.writer = csv.DictWriter(f, dct.keys())
                self.writer.writerows(self.rows)

                f.flush()
                del self.rows[:]

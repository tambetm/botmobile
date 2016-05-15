import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument("csv_file")
parser.add_argument("--png_file")
parser.add_argument("--dpi", type = int, default = 80)
parser.add_argument("--skiprows", type = int, default = 1)
parser.add_argument("--delimiter", default = ",")
parser.add_argument("--fields", type = lambda s: [item for item in s.split(',')], default = "distRaced,epsilon,avgmaxQ,avgloss")
parser.add_argument("--figure_width", type = int, default = 16)
parser.add_argument("--figure_height", type = int, default = 9)
args = parser.parse_args()

# field definitions for numpy
dtype = [
  ("episode", "int"),
  ("distFromStart", "float"),
  ("distRaced", "float"),
  ("curLapTime", "float"),
  ("lastLapTime", "float"),
  ("racePos", "int"),
  ("epsilon", "float"),
  ("replay_memory", "int"),
  ("train_steps", "int"),
  ("avgmaxQ", "float"),
  ("avgloss", "float")
]
data = np.loadtxt(args.csv_file, skiprows = args.skiprows, delimiter = args.delimiter, dtype = dtype)

# definitions for plot titles
labels = {
  "episode": "Episode", 
  "distFromStart": "Distance from start",
  "distRaced": "Distance raced",
  "curLapTime": "Current lap time",
  "lastLapTime": "Last lap time",
  "racePos": "Race position",
  "epsilon": "Exploration epsilon",
  "replay_memory": "Replay memory",
  "train_steps": "Train steps",
  "avgmaxQ": "Average max Q-value",
  "avgloss": "Average loss"
}

# calculate number of subplots
nr_fields = len(args.fields)
cols = math.ceil(math.sqrt(nr_fields))
rows = math.ceil(nr_fields / float(cols))

plt.figure(figsize = (args.figure_width, args.figure_height))

# plot all fields
for i, field in enumerate(args.fields):
  plt.subplot(rows, cols, i + 1)

  plt.plot(data['episode'], data[field])
  plt.ylabel(labels[field])
  plt.xlabel(labels['episode'])
  plt.title(labels[field])

plt.tight_layout()

# if png_file argument given, save to file
if args.png_file is not None:
  plt.savefig(args.png_file, dpi = args.dpi)
else:
  plt.show()

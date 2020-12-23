from AnalysingFunctions import *

from FileHandling import *
import time
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from natsort import natsorted

sns.set_theme(style='whitegrid')
pio.renderers.default = 'browser'
subject = 1
env = 'W'
target =0
block = 1
output = read_hololens_data(target=target, environment=env, posture='W', block=block, subject=subject,
                            study_num=3)

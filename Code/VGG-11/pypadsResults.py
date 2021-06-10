from pypads.app import base, results
import pandas as pd
import numpy as np
import os
import json

os.environ['MONGO_DB'] = "pypads"

os.environ['MONGO_USER'] = "pypads"

os.environ['MONGO_URL'] = "mongodb://www.padre-lab.eu:2222"

os.environ['MONGO_PW'] = "8CN7OqknwhYr3RO"


tracker = base.PyPads(autostart=True)
obj1 = results.PyPadsResults()

res = obj1.get_experiment(experiment_name="Effect of GPUs - Logistic map")
print('Exp details', res)
#experiment_ids='7'

res2 = obj1.get_experiments_data_frame(experiment_names="Effect of GPUs - Logistic map")
print('Output', res2)
df = pd.DataFrame(res2)
print('DF ', df)

np.savetxt(r'np2.txt', df.values, fmt='%s')

from pypads.app import base, results
import pandas as pd
import numpy as np
import os
#os.system("export MONGO_DB = pypads")
#os.system("export MONGO_USER = pypads")
#os.system("export MONGO_URL = 'mongodb://www.padre-lab.eu:2222'")
#os.system("export MONGO_PW = 8CN7OqknwhYr3RO")

os.environ['MONGO_DB'] = "pypads"

os.environ['MONGO_USER'] = "pypads"

os.environ['MONGO_URL'] = "mongodb://www.padre-lab.eu:2222"

os.environ['MONGO_PW'] = "8CN7OqknwhYr3RO"

tracker = base.PyPads(autostart=True)
obj1 = results.PyPadsResults()

#obj1.get_tracked_objects()
res = obj1.get_experiment(experiment_name="Effect of GPUs - VGG 11")
print('Exp details', res)

res2 = obj1.get_experiments_data_frame(experiment_names="Effect of GPUs - VGG 11", experiment_ids='8')
print('Output', res2)
res2 = res2.values[0]

data1 = pd.DataFrame(res)
print('dataFrame', data1)

mask = np.column_stack([res2[col].str.contains(r".rgb", na=False) for col in res2])
print('mask ', res2.loc[mask.any(axis=1)])
# review maker b code 
## 20261015  

### 依赖关系
依赖scenarios,counterfactual_eval,models,rl_env,rl_bidding,rl_td3,run_artifacts,surplus_metrics.

### known modules
time -- to record time
random -- to get random number
datetime -- to get time and day
numpy -- an important and classic module to handle statics
torch -- used in machine learning and training

from scanarios import get_scenarios -- get set scenarios
from models import MarketConfig -- get basic marker congfig
from rl_bidding import agent_action_spaces -- get action space


### unknown modules 
os,argparse,copy
from counterfactual_eval import validation_kpis
import day_rotation
from rl_env import BiddingEnv, DayCapture, BLOCK_SIZE, N_BLOCKS
from rl_td3 import CALIBRATED_DEV_PENALTY, TD3, save_policy
from run_artifacts import RunArtifacts
from surplus_metrics import day_econ_metrics
### 
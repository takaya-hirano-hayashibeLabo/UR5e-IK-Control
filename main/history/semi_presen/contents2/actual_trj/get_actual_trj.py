import sys
from pathlib import Path
import argparse
ROOT=Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.append(str(ROOT))
print("ROOT PATH:",ROOT)

import os
import pandas as pd
import yaml
from src.trajectory_generator import generate_trajectory



def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--configpath",required=True)
    parser.add_argument("--timescale",default=1.0,type=float)
    parser.add_argument("--saveto",required=True)
    args=parser.parse_args()

    config_path=args.configpath
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    config["trajectory"]["loop_duration"]*=args.timescale
    trajectory=generate_trajectory(config["trajectory"])

    trajectory_db=pd.DataFrame(trajectory,columns=["time","target_x","target_y"])

    result_dir=Path(args.saveto)/f"trajectory_timescale{args.timescale}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    trajectory_db.to_csv(result_dir/"trajectory.csv",index=False)

    args_dict=vars(args)
    out_dict={
        "args":args_dict,
        "config":config["trajectory"]
    }
    with open(result_dir/"args.yml",'w') as f:
        yaml.dump(out_dict,f,indent=4)

if __name__=="__main__":
    main()



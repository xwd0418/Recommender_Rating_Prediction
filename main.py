import yaml, sys, os
import numpy as np, random
from experiment import Experiment

if __name__ == "__main__":
    
    seed = 10086
    np.random.seed(seed)
    random.seed(seed)
    
    if len(sys.argv) > 2:
        config_path = sys.argv[1]
        
    else:
        raise Exception(" missing config_path")
    
    # load config file
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    # creating experiment folder       
    config_name =  config_path.split('/')[-1].split('.')[0]
    output_path = f"experiment_results/{config_name}/"
    os.makedirs(output_path, exist_ok = True)
    
    # make a copy of config
    with open(f'{output_path}config_copy.yml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
        
    config['file_output_dir_path'] = output_path
    
    exp = Experiment(config)
    exp.train()
    exp.test() 
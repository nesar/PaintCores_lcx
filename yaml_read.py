import yaml

def process(**params):    # pass in variable numbers of args
    for key, value in params.items():
        print('%s: %s' % (key, value))

def yaml_config(fileIn):
    
    with open(fileIn, 'rb') as f:
        config = yaml.safe_load(f.read())    # load the config file
            
    return config

    # process(**config)    # pass in your keyword args
import yaml

config_application = "config/config_main.yaml"
config_server = "config/config_server.yaml"
config_client = "config/config_client.yaml"
config_experiment = "config/config_experiment.yaml"
config_main= "config/config_main.yaml"

def read_server_config():
    with open(config_server, 'r') as f:
        config = yaml.safe_load(f)

    f.close()

    # return config[config["experiment_to_conduct"]]
    return config

def read_config(experiment):
    with open(experiment, 'r') as f:
        config = yaml.safe_load(f)

    f.close()
    
    return config

def read_client_config():
    with open(config_client) as f:
        config = yaml.safe_load(f)

    f.close()

    return config

def read_application_config():
    with open(config_application) as f:
        config = yaml.safe_load(f)

    f.close()

    return config

def read_experiment_config():
    with open(config_experiment) as f:
        config = yaml.safe_load(f)

    f.close()

    return config

def change_experiment(experiment):
    with open(config_server) as f:
        doc = yaml.safe_load(f)
    f.close()

    doc['experiment_to_conduct'] = experiment

    with open(config_server, 'w') as f:
        yaml.dump(doc, f)
    f.close()
    
def get_experiments(experiment=None):
    with open(config_main) as f:
        doc = yaml.safe_load(f)
    f.close()

    if doc['based'] == "models":
        experiments = doc['experiments'][doc['based']][doc['run']]

    return experiments

def change_experiments(levels):
    with open(config_server) as f:
        doc = yaml.safe_load(f)
    f.close()

    doc['experiment'] = levels

    with open(config_server, 'w') as f:
        yaml.dump(doc, f)
    f.close()
        
def change_experiment_layer(layer):
    with open(config_experiment) as f:
        doc = yaml.safe_load(f)
    f.close()

    doc['experiment']['conv']['layer'] = layer

    with open(config_experiment, 'w') as f:
        yaml.dump(doc, f)
    f.close()    
        
def change_SL_layer(layer):
    with open(config_server) as f:
        doc = yaml.safe_load(f)
    f.close()
    doc['SL_config']['split_layer']['AlexNet']["client"]["conv1"] = False
    doc['SL_config']['split_layer']['AlexNet']["client"][f"conv{layer}"] = True
    
    doc['SL_config']['split_layer']['AlexNet']["server"]["conv2"] = False
    doc['SL_config']['split_layer']['AlexNet']["server"][f"conv{layer+1}"] = True

    with open(config_server, 'w') as f:
        yaml.dump(doc, f)
    f.close()
        
def change_iid_to_niid(setting, dataset):
    with open(config_server) as f:
        doc = yaml.safe_load(f)
    f.close()
    
    doc["collection"]["datasets"][dataset]["iid"] = True if setting =="iid" else False
    
    with open(config_server, 'w') as f:
        yaml.dump(doc, f)
    
    f.close()

def change_aggregator(aggregator, learning):
    with open(config_server) as f:
        doc = yaml.safe_load(f)
    f.close()
    
    where = 'FL_config' if learning== 'FL' else 'SL_config'

    doc[where]["aggregator"] = aggregator

    with open(config_server, 'w') as f:
        yaml.dump(doc, f)
    f.close()

def change_personalised_learning():
    with open(config_server) as f:
        doc = yaml.safe_load(f)
    f.close()
    
    where = 'PFL'

    doc["configuration"] = where

    with open(config_server, 'w') as f:
        yaml.dump(doc, f)
    f.close()

def change_personalised_aggregator(aggregator):
    with open(config_server) as f:
        doc = yaml.safe_load(f)
    f.close()
    
    where = 'PFL_config'

    doc[where]["aggregator"] = aggregator

    with open(config_server, 'w') as f:
        yaml.dump(doc, f)
    f.close()

def change_learning(learning):
    with open(config_server) as f:
        doc = yaml.safe_load(f)
    f.close()

    doc['configuration'] = learning

    with open(config_server, 'w') as f:
        yaml.dump(doc, f)
    f.close()
        
def change_dataset(dataset):
    with open(config_server) as f:
        doc = yaml.safe_load(f)
    f.close()
    
    doc["collection"]["selection"] = dataset
    
    with open(config_server, 'w') as f:
        yaml.dump(doc, f)
    
    f.close()
        
def change_number_of_clients(clients, learning):
    with open(config_server) as f:
        doc = yaml.safe_load(f)
    f.close()
    
    where = 'FL_config' if learning== 'FL' else 'SL_config'
    
    doc[where]["K"] = clients
    doc["defaults"]["K"] = clients
    
    with open(config_server, 'w') as f:
        yaml.dump(doc, f)
    
    f.close()
    
def change_numbers_of_adversaries(number):

    with open(config_client) as f:
        doc = yaml.safe_load(f)
    f.close()
    
    for client in doc["clients"]:
        if type(client) == int:
            doc["clients"][client]["adversary"] = False

    for num in range(1,number+1):
        num = num*2 - 1

        doc["clients"][num]['adversary'] = True

    with open(config_client, 'w') as f:
        yaml.dump(doc, f)
    
    f.close()

def modify_experiments_confidence(levels):
    with open(config_experiment) as f:
        doc = yaml.safe_load(f)
    f.close()

    doc['target_confidence'] = levels

    with open(config_experiment, 'w') as f:
        yaml.dump(doc, f)
    f.close()

def add_more_adversaries(number):

    with open(config_client) as f:
        doc = yaml.safe_load(f)
    f.close()

    for client_id in range(11,100):
        if type(client_id) == int:
            new_row = doc["clients"][2]
            doc["clients"][client_id] = new_row

    for num in range(1,number+1):
        num = num*2 - 1

        doc["clients"][num]['adversary'] = True

    with open(config_client, 'w') as f:
        yaml.dump(doc, f)
    f.close()

if __name__ =="__main__":
    print("test")
from src.configLibrary import (
    read_application_config,
    read_client_config,
    read_experiment_config,
    read_server_config,
    read_config,
)

server_configuration = read_server_config()

def init(experiment):
    global server_configuration
    
    server_configuration = read_config(experiment)

client_configuration = read_client_config()


import config

configuration = config.server_configuration
prefix = "{}_".format(configuration["configuration"])
aggregation = configuration["learning_config"]["aggregator"]

if configuration['configuration'] == 'FL':
    
    if aggregation == "FedAvg":
        from src.learning.server.serverFedAvg import ServerFedAvg as Server
    elif aggregation == "Krum":
        from src.learning.server.serverKrum import ServerKrum as Server
    else:
        raise NotImplementedError(f'This aggregation {aggregation} is not implemented. Please check the config_server.yaml')
        
elif configuration['configuration'] == 'PFL':
    if aggregation == "FedProx":
        from src.learning.server.serverFedProx import ServerFedProx as Server
    elif aggregation == "PerFedAvg":
        from src.learning.server.serverPerFedAvg import ServerPerFedAvg as Server
    elif aggregation == "FedDyn":
        from src.learning.server.serverFedDyn import ServerFedDyn as Server
    elif aggregation == "Ditto":
        from src.learning.server.serverDitto import ServerDitto as Server
    elif aggregation == "pFedME":
        from src.learning.server.serverPFedME import ServerPFedME as Server
    else:
        raise NotImplementedError(f'This aggregation {aggregation} is not implemented. Please check the config_server.yaml')
    
elif configuration['configuration'] == 'SL':
    raise NotImplementedError(f'This learning algorithm is not implemented. Please check the config_server.yaml')

elif configuration["configuration"] == "CML":
    from src.learning.server.serverCentralisedML import ServerCentralisedML as Server
    
else:
    raise AttributeError(f'Please check the config_server.yaml, wrong configuration specification')

__all__ = [
    "Server"
]
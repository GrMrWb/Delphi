import git
import logging

logger = logging.getLogger(__name__)

def commit_to_git():
    repo = git.Repo('.')
    repo.git.add(logger.root.handlers[0].baseFilename)
    repo.git.commit('-m', 'Application Log for remote access', author='m.aristodemou@lboro.ac.uk')
    origin = repo.remote(name='origin')
    origin.push()
    
def commit_results(num_of_clients, path):
    repo = git.Repo('.')
    
    # Clients
    attributes = ["validation", "testing", "training"]
    for attribute in attributes:
        for i in range(1, num_of_clients+1):
            repo.git.add(f"{path}/Client_{i}_{attribute}_performance.csv")
 
    #Server
    attributes = ["validation", "testing"]
    for attribute in attributes:
        repo.git.add(f"{path}/Server_{attribute}_performance.csv")
    
    # Epoch Number
    repo.git.add(f"{path}/epoch.txt")
    
    repo.git.commit('-m', 'Models checkpoint', author='m.aristodemou@lboro.ac.uk')
    origin = repo.remote(name='origin')
    origin.push()
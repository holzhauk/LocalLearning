import os
import torch
from pathlib import Path
import LocalLearning_copy as LocalLearning

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def load_trained_model_bp(idx):
    trained_model_bp_path = Path("../data/models/KHModelCIFAR10_ensemble/bp")
    file_names_trained_bp = os.listdir(trained_model_bp_path)
    file_names_trained_bp = [fn for fn in file_names_trained_bp if os.path.isfile(trained_model_bp_path / Path(fn))]
    
    trained_model_bp = Path(file_names_trained_bp[idx])
    
    with torch.no_grad():
        trained_state_bp = torch.load(trained_model_bp_path/trained_model_bp)
        model_ps_bp = trained_state_bp["fkhl3-state"]
        model_bp = LocalLearning.KHModel_bp(model_ps_bp)
        model_bp.eval()
        model_bp.load_state_dict(trained_state_bp["model_state_dict"])
        model_bp.to(device)
        
    return model_bp

def load_trained_model_ll(idx):
        
    trained_model_ll_path = Path("../data/models/KHModelCIFAR10_ensemble/ll")
    file_names_trained_ll = os.listdir(trained_model_ll_path)
    file_names_trained_ll = [fn for fn in file_names_trained_ll if os.path.isfile(trained_model_ll_path / Path(fn))]
    
    trained_model_ll = Path(file_names_trained_ll[idx])
    
    with torch.no_grad():
        trained_state_ll = torch.load(trained_model_ll_path/trained_model_ll)
        model_ps_ll = trained_state_ll["fkhl3-state"]
        model_ll = LocalLearning.KHModel(model_ps_ll)
        model_ll.eval()
        model_ll.load_state_dict(trained_state_ll["model_state_dict"])
        model_ll.to(device)
        
    return model_ll

def load_Konstantin_model(file_name, model_type):
    trained_model_path = Path("../data/models/KHModelCIFAR10_ensemble")
    
    if model_type == "bp": 
        modeltype = LocalLearning.KHModel_bp
    
    elif model_type == "ll":
        modeltype = LocalLearning.KHModel

    with torch.no_grad():
        trained_state = torch.load(trained_model_path/file_name)
        model_ps = trained_state["fkhl3-state"]
        model = modeltype(model_ps)
        model.eval()
        model.load_state_dict(trained_state["model_state_dict"])
        model.to(device)
        
    return model
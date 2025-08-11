"""Contains training and predictions functions for Chemprop models."""
#from pathlib import Path # hth
import numpy as np
#from chemprop.models import MoleculeModel
from ppsci.arch.chemprop_molecule import MoleculeModel # hth
#from chemprop.utils import load_checkpoint, load_scalers # hth
from sklearn.preprocessing import StandardScaler

'''
def chemprop_load(model_path: Path, device: (paddle.CPUPlace, paddle.
    CUDAPlace, str)=paddle.CPUPlace()) ->MoleculeModel:
    """Loads a Chemprop model.

    :param model_path: A path to a Chemprop model.
    :param device: The device on which to load the model.
    :return: A Chemprop model.
    """
    return load_checkpoint(path=str(model_path), device=device)


def chemprop_load_scaler(model_path: Path) ->StandardScaler:
    """Loads a Chemprop model's data scaler.

    :param model_path: A path to a Chemprop model.
    :return: A data scaler.
    """
    return load_scalers(path=str(model_path))[0]
''' # hth


def chemprop_predict_on_molecule(model: MoleculeModel, smiles: str,
    fingerprint: (np.ndarray)=None, scaler: (StandardScaler)=None
    ) ->float:
    """Predicts the property of a molecule using a Chemprop model.

    :param model: A Chemprop model.
    :param smiles: A SMILES string.
    :param fingerprint: A 1D array of molecular fingerprints (if applicable).
    :param scaler: A data scaler (if applicable).
    :return: The prediction on the molecule.
    """

    #generator 单独运行, reaction_to_building_blocks_filtered.pkl
    
    from synthemol.features import BatchMolGraph, mol2graph
    batch = [mol2graph(b) for b in [[smiles]]]
    batchs = [batch[0].get_components()]
    #print(len(batchs))
    #print("smiles",batchs,"/r/nfingerprint", fingerprint)
    pred = model(batch=batchs, features_batch=[fingerprint] if 
        fingerprint is not None else None).item()

    #pred = model(batch=[[smiles]], features_batch=[fingerprint] if 
    #    fingerprint is not None else None).item()

    if scaler is not None:
        pred = scaler.inverse_transform([[pred]])[0][0]
    return float(pred)


def chemprop_predict_on_molecule_ensemble(models: list[MoleculeModel],
    smiles: str, fingerprint: (np.ndarray)=None, scalers: (list[
    StandardScaler])=None) ->float:
    """Predicts the property of a molecule using an ensemble of Chemprop models.

    :param models: An ensemble of Chemprop models.
    :param smiles: A SMILES string.
    :param fingerprint: A 1D array of molecular fingerprints (if applicable).
    :param scalers: An ensemble of data scalers (if applicable).
    :return: The ensemble prediction on the molecule.
    """
    return float(np.mean([chemprop_predict_on_molecule(model=model, smiles=
        smiles, fingerprint=fingerprint, scaler=scaler) for model, scaler in
        zip(models, scalers)]))

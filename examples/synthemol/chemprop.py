# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import hydra
import numpy as np
#from ednn_utils import Meter
from omegaconf import DictConfig

import paddle
#from typing import Callable
from args import TrainArgs
from loss_functions import get_loss_func

import ppsci
import ppsci.arch.chemprop_molecule
#from ppsci.utils import logger

def get_train_loss_func(args, pos_weights=None):  #:paddle.Tensor=None):
    def train_loss_func(output_dict, label_dict, weight_dict):
        '''
        if reg:
            loss_func = MSELoss(reduction="none")
        else:
            loss_func = BCEWithLogitsLoss(reduction="none", pos_weight=pos_weights)
        return {
            "pred": (
                loss_func(output_dict["pred"], label_dict["y"])
                * (label_dict["mask"] != 0).astype("float32")
            ).mean()
        }
        '''
        #print(len(batch), args.loss_function, args.dataset_type)

        preds = output_dict['pred']

        targets = label_dict['targets']
        target_weights = label_dict['target_weights']
        data_weights = label_dict['data_weights']
        mask = label_dict['mask']

        loss_func = get_loss_func(args)
        if args.loss_function == 'bounded_mse':
            lt_target_batch = lt_target_batch #.to(torch_device)
            gt_target_batch = gt_target_batch #.to(torch_device)
        if (args.loss_function == 'mcc' and args.dataset_type ==
            'classification'):
            loss = loss_func(preds, targets, data_weights, mask
                ) * target_weights.squeeze(axis=0)
        elif args.loss_function == 'mcc':
            targets = targets.astype(dtype='int64')
            target_losses = []
            for target_index in range(preds.shape[1]):
                target_loss = loss_func(preds[:, target_index, :], targets[
                    :, target_index], data_weights, mask[:, target_index]
                    ).unsqueeze(axis=0)
                target_losses.append(target_loss)
            #loss = paddle.concat(x=target_losses).to(torch_device
            #    ) * target_weights.squeeze(axis=0)
            loss = paddle.concat(x=target_losses) * target_weights.squeeze(axis=0)
        elif args.dataset_type == 'multiclass':
            targets = targets.astype(dtype='int64')
            if args.loss_function == 'dirichlet':
                loss = loss_func(preds, targets, args.evidential_regularization
                    ) * target_weights * data_weights * mask
            else:
                target_losses = []
                for target_index in range(preds.shape[1]):
                    target_loss = loss_func(preds[:, target_index, :],
                        targets[:, target_index]).unsqueeze(axis=1)
                    target_losses.append(target_loss)
                #loss = paddle.concat(x=target_losses, axis=1).to(torch_device
                #    ) * target_weights * data_weights * mask
                loss = paddle.concat(x=target_losses, axis=1) * target_weights * data_weights * mask
        elif args.dataset_type == 'spectra':
            loss = loss_func(preds, targets, mask
                ) * target_weights * data_weights * mask
        elif args.loss_function == 'bounded_mse':
            loss = loss_func(preds, targets, lt_target_batch, gt_target_batch
                ) * target_weights * data_weights * mask
        elif args.loss_function == 'evidential':
            loss = loss_func(preds, targets, args.evidential_regularization
                ) * target_weights * data_weights * mask
        elif args.loss_function == 'dirichlet':
            loss = loss_func(preds, targets, args.evidential_regularization
                ) * target_weights * data_weights * mask
        else:
            loss = loss_func(preds, targets
                ) * target_weights * data_weights * mask
        loss = loss.sum() / mask.sum()

        return {
            "pred": loss.astype("float32")
        }

    return train_loss_func


def get_val_loss_func(reg, metric):
    def val_loss_func(output_dict, label_dict):
        eval_metric = Meter()
        eval_metric.update(output_dict["pred"], label_dict["y"], label_dict["mask"])

        if reg:
            rmse_score = np.mean(eval_metric.compute_metric(metric))
            mae_score = np.mean(eval_metric.compute_metric("mae"))
            r2_score = np.mean(eval_metric.compute_metric("r2"))
            return {"rmse": rmse_score, "mae": mae_score, "r2": r2_score}
        else:
            roc_score = np.mean(eval_metric.compute_metric(metric))
            prc_score = np.mean(eval_metric.compute_metric("prc_auc"))
            return {"roc_auc": roc_score, "prc_auc": prc_score}

    return val_loss_func

def make_args(dataset_type, epochs, use_gpu, fingerprint_type, property_name):
    # , train_smiles, train_fingerprints):

    # Create args
    arg_list = [
        '--data_path', 'foo.csv',
        '--dataset_type', dataset_type,
        '--save_dir', 'foo',
        '--epochs', str(epochs),
        '--quiet'
    ] + ([] if use_gpu else ['--no_cuda'])

    if fingerprint_type =='morgan':
        arg_list += ['--features_generator', 'morgan']
    elif fingerprint_type == 'rdkit':
        arg_list += ['--features_generator', 'rdkit_2d_normalized', '--no_features_scaling']
    elif fingerprint_type == None:
        pass
    else:
        raise ValueError(f'Fingerprint type "{fingerprint_type}" is not supported.')

    args = TrainArgs().parse_args(arg_list)
    args.task_names = [property_name]
    args.train_data_size = 100 #len(train_smiles) # TODO: magic

    if fingerprint_type is not None:
        args.features_size = 128 #train_fingerprints.shape[1] # TODO: magic
    return args

def load_raw_data():
    import pandas as pd
    from random import Random
    data_path = './data/Data/1_training_data/antibiotics_hits.csv'
    data = pd.read_csv(data_path)
    print(f'Data size = {len(data):,}')
    num_models = 10
    num_folds = 10
    indices = np.tile(np.arange(num_folds), 1 + len(data) // num_folds)[:
        len(data)]
    random = Random(0)
    random.shuffle(indices)
    assert 1 <= num_models <= num_folds
    smiles_column = 'smiles'
    property_column = 'antibiotic_activity'
    
    #for model_num in trange(num_models, desc='cross-val'):
    #print(f'Model {model_num}')
    model_num = 1 # TODO: magic
    test_index = model_num
    val_index = (model_num + 1) % num_folds
    test_mask = indices == test_index
    val_mask = indices == val_index
    train_mask = ~(test_mask | val_mask)
    test_data = data[test_mask]
    val_data = data[val_mask]
    train_data = data[train_mask]
    print("test_data:",len(test_data),"train_data:",len(train_data),"val_data:",len(val_data))
    train_smiles=train_data[smiles_column]
    train_fingerprints=None
    train_properties=train_data[property_column]
    return train_smiles, train_fingerprints, train_properties

def train(cfg: DictConfig):

    train_smiles, train_fingerprints, train_properties = load_raw_data()

    args = make_args(
        dataset_type="classification",
        epochs=1,
        use_gpu=True,
        fingerprint_type=None,
        property_name="antibiotic_activity"
    )
    
    # set dataloader config
    train_dataloader_cfg = {
        "dataset": {
            "name": "MoleculeDatasetIter",
            "input_keys": ("mol_batch", "features_batch", "atom_descriptors_batch",
                "atom_features_batch", "bond_features_batch"),
            "args": args,
            "smiles": train_smiles,
            "fingerprints": train_fingerprints,
            "properties": train_properties,
            "label_keys": (
                "targets",
                "data_weights",
                "mask",
                "target_weights",
            ),
            #"data_dir": cfg.data_dir,
            #"data_mode": "train",
            #"data_label": cfg.data_label,
        },
        #"batch_size": cfg.TRAIN.batch_size,
        #"sampler": {
        #    "name": "BatchSampler",
        #    "drop_last": False,
        #    "shuffle": True,
        #},
        "num_workers": 1,
    }

    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        train_dataloader_cfg,
        output_expr={"pred": lambda out: out["pred"]},
        loss=ppsci.loss.FunctionalLoss(get_train_loss_func(args)),
        name="Sup",
    )

    # params from dataset
    '''
    inputs = sup_constraint.data_loader.dataset.data_tr_x.shape[1]
    tasks = sup_constraint.data_loader.dataset.task_dict[cfg.data_label]
    iters_per_epoch = len(sup_constraint.data_loader)
    logger.info(f"inputs is: {inputs}, iters_per_epoch: {iters_per_epoch}")
    if not reg:
        pos_weights = sup_constraint.data_loader.dataset.pos_weights
        sup_constraint.loss = ppsci.loss.FunctionalLoss(
            get_train_loss_func(reg, pos_weights)
        )
    '''

    # wrap constraints together
    constraint = {sup_constraint.name: sup_constraint}

    '''
    hyper_paras = cfg.HYPER_OPT[cfg.data_label]

    hidden_units = [
        hyper_paras["hidden_unit1"],
        hyper_paras["hidden_unit2"],
        hyper_paras["hidden_unit3"],
    ]
    '''


    # set model
    '''
        # **cfg.MODEL,
        input_keys=("x",),
        output_keys=("pred",),
        hidden_units=hidden_units,
        embed_name=cfg.MODEL.embed_name,
        inputs=inputs,
        outputs=len(tasks),
        d_out=hyper_paras["d_out"],
        sigma=hyper_paras["sigma"],
        dp_ratio=hyper_paras["dropout"],
        reg=reg,
        first_omega_0=hyper_paras["omega0"],
        hidden_omega_0=hyper_paras["omega1"],
    '''
    model = ppsci.arch.chemprop_molecule.MoleculeModel(
        args=args
    )

    # set optimizer
    #weight_decay=hyper_paras["l2"]
    # TODO: magic weight_decay
    optimizer = ppsci.optimizer.Adam(
        learning_rate=cfg.TRAIN.learning_rate, weight_decay=0.001 
    )(model)

    '''
    # set validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "IFMMoeDataset",
            "input_keys": ("x",),
            "label_keys": (
                "y",
                "mask",
            ),
            "data_dir": cfg.data_dir,
            "data_mode": "val",
            "data_label": cfg.data_label,
        },
        "batch_size": cfg.EVAL.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "num_workers": 1,
    }

    rmse_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(get_train_loss_func(reg)),
        output_expr={"pred": lambda out: out["pred"]},
        metric={
            "MyMeter": ppsci.metric.FunctionalMetric(get_val_loss_func(reg, metric))
        },
        name="MyMeter_validator",
    )
    if not reg:
        pos_weights = rmse_validator.data_loader.dataset.pos_weights
        rmse_validator.loss = ppsci.loss.FunctionalLoss(
            get_train_loss_func(reg, pos_weights)
        )

    validator = {rmse_validator.name: rmse_validator}
    '''

    # initialize solver
    solver = ppsci.solver.Solver(
        model,
        constraint,
        cfg.output_dir,
        optimizer,
        None,
        cfg.HYPER_OPT[cfg.data_label].epoch,  # cfg.TRAIN.epochs,
        2 ,#iters_per_epoch, # TODO: magic
        save_freq=cfg.TRAIN.save_freq,
        eval_during_train=cfg.TRAIN.eval_during_train,
        eval_freq=cfg.TRAIN.eval_freq,
        #validator=validator,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
        checkpoint_path=cfg.TRAIN.checkpoint_path,
    )

    # train model
    solver.train()

import pandas as pd
from chemprop_models import chemprop_predict, my_chemprop_load
from tqdm import tqdm
from pathlib import Path

def pre_compute():
    pass
    data_path = Path("./data/Data/4_real_space/building_blocks.csv")
    model_path = Path("./outputs_ifm/doc_metric/checkpoints/epoch_44.pdparams")
    smiles_column = 'smiles'
    model_type = 'chemprop'
    fingerprint_type = None
    use_gpu = True
    average_preds = True
    num_workers = 1
    preds_column_prefix = None
    save_path = Path('./pre-compute-hth/building_blocks.csv')


    args = make_args(
        dataset_type="classification",
        epochs=1,
        use_gpu=True,
        fingerprint_type=None,
        property_name="antibiotic_activity"
    )

    model = ppsci.arch.chemprop_molecule.MoleculeModel(
        args=args
    )
    
    data = pd.read_csv(data_path)
    smiles = list(data[smiles_column])
    if model_type != 'chemprop' and fingerprint_type is None:
        raise ValueError('Must define fingerprint_type if using sklearn model.'
            )
    if fingerprint_type is not None:
        #fingerprints = compute_fingerprints(smiles, fingerprint_type=
        #    fingerprint_type)
        pass
    else:
        fingerprints = None
    if model_path.is_dir():
        model_paths = list(model_path.glob('**/*.pt' if model_type ==
            'chemprop' else '**/*.pkl'))
        if len(model_paths) == 0:
            raise ValueError(
                f'Could not find any models in directory {model_path}.')
    else:
        model_paths = [model_path]
    if model_type == 'chemprop':
        if use_gpu:
            device = str('cuda').replace('cuda', 'gpu')
        else:
            device = paddle.CPUPlace()
        paddle.seed(seed=0)
        #models = [chemprop_load(model_path=model_path, device=device) for
        #    model_path in model_paths]
        models = [my_chemprop_load(model, model_path=model_path, device=device) for
            model_path in model_paths]

    



    #else:
    #    models = [sklearn_load(model_path=model_path) for model_path in
    #        model_paths]
    print(model_paths, models)

    if model_type == 'chemprop':
        preds = np.array([chemprop_predict(model=m, smiles=smiles,
            fingerprints=fingerprints, num_workers=num_workers) for m in
            tqdm(models, desc='models')])
    #else:
    #    preds = np.array([sklearn_predict(model=model, fingerprints=
    #        fingerprints) for model in tqdm(models, desc='models')])

    if average_preds:
        preds = np.mean(preds, axis=0)
    model_string = (
        f"{model_type}{f'_{fingerprint_type}' if fingerprint_type is not None else ''}"
        )
    preds_string = (
        f"{f'{preds_column_prefix}_' if preds_column_prefix is not None else ''}{model_string}"
        )
    if average_preds:
        data[f'{preds_string}_ensemble_preds'] = preds
    else:
        for model_num, model_preds in enumerate(preds):
            data[f'{preds_string}_model_{model_num}_preds'] = model_preds
    if save_path is None:
        save_path = data_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(save_path, index=False)


from datetime import datetime
from pathlib import Path
import pandas as pd
#from examples.chemprop.synthemol.constants import BUILDING_BLOCKS_PATH, FINGERPRINT_TYPES, MODEL_TYPES, OPTIMIZATION_TYPES, REACTION_TO_BUILDING_BLOCKS_PATH, REAL_BUILDING_BLOCK_ID_COL, SCORE_COL, SMILES_COL
from examples.synthemol.synthemol.reactions import Reaction, REACTIONS, load_and_set_allowed_reaction_building_blocks, set_all_building_blocks
from examples.synthemol.synthemol.generate.generator import Generator
from examples.synthemol.synthemol.generate.utils import create_model_scoring_fn, save_generated_molecules

def generate():
    pass

    
    model_path = None 
    model_type = 'chemprop' #: MODEL_TYPES
    save_dir = Path('./hth_generated') #: Path,
    
    building_blocks_path = "./pre-compute-hth/building_blocks.csv" #: Path=BUILDING_BLOCKS_PATH
    fingerprint_type = None #: (FINGERPRINT_TYPES)=None
    reaction_to_building_blocks_path = './data/Data/4_real_space/reaction_to_building_blocks_filtered.pkl' #: (Path)=  REACTION_TO_BUILDING_BLOCKS_PATH
    building_blocks_id_column = 'Reagent_ID' #: str= REAL_BUILDING_BLOCK_ID_COL
    building_blocks_score_column = 'chemprop_ensemble_preds' # : str=SCORE_COL
    
    building_blocks_smiles_column = 'smiles' #: str=SMILES_COL
    reactions = REACTIONS #: tuple[Reaction]=REACTIONS
    max_reactions = 1 #: int=1
    n_rollout = 20 #20000 #: int=10
    
    explore_weight = 10.0 #: float=10.0
    num_expand_nodes = None #: (int)=None
    
    optimization = 'maximize' #: OPTIMIZATION_TYPES='maximize'
    rng_seed = 0 #: int=0
    
    no_building_block_diversity = False #: bool=False
    store_nodes = False #: bool=False
    
    verbose = False #: bool=False
    replicate = True #: bool=False




    save_dir.mkdir(parents=True, exist_ok=True)
    print('Loading building blocks...')
    if replicate:
        building_block_data = pd.read_csv(building_blocks_path, dtype={
            building_blocks_score_column: str})
        building_block_data[building_blocks_score_column
            ] = building_block_data[building_blocks_score_column].astype(float)
        old_reactions_order = [275592, 22, 11, 527, 2430, 2708, 240690, 
            2230, 2718, 40, 1458, 271948, 27]
        reactions = tuple(sorted(reactions, key=lambda reaction:
            old_reactions_order.index(reaction.id)))
        building_block_data.drop_duplicates(subset=
            building_blocks_smiles_column, inplace=True)
    else:
        building_block_data = pd.read_csv(building_blocks_path)
    print(f'Loaded {len(building_block_data):,} building blocks')
    if building_block_data[building_blocks_id_column].nunique() != len(
        building_block_data):
        raise ValueError('Building block IDs are not unique.')
    building_block_smiles_to_id = dict(zip(building_block_data[
        building_blocks_smiles_column], building_block_data[
        building_blocks_id_column]))
    building_block_id_to_smiles = dict(zip(building_block_data[
        building_blocks_id_column], building_block_data[
        building_blocks_smiles_column]))
    building_block_smiles_to_score = dict(zip(building_block_data[
        building_blocks_smiles_column], building_block_data[
        building_blocks_score_column]))
    print(f'Found {len(building_block_smiles_to_id):,} unique building blocks')
    set_all_building_blocks(reactions=reactions, building_blocks=set(
        building_block_smiles_to_id))
    if reaction_to_building_blocks_path is not None:
        print(
            'Loading and setting allowed building blocks for each reaction...')
        load_and_set_allowed_reaction_building_blocks(reactions=reactions,
            reaction_to_reactant_to_building_blocks_path=
            reaction_to_building_blocks_path)
    print('Loading models and creating model scoring function...')
    model_scoring_fn = create_model_scoring_fn(model_path=model_path,
        model_type=model_type, fingerprint_type=fingerprint_type,
        smiles_to_score=building_block_smiles_to_score)
    print('Setting up generator...')
    generator = Generator(building_block_smiles_to_id=
        building_block_smiles_to_id, max_reactions=max_reactions,
        scoring_fn=model_scoring_fn, explore_weight=explore_weight,
        num_expand_nodes=num_expand_nodes, optimization=optimization,
        reactions=reactions, rng_seed=rng_seed, no_building_block_diversity
        =no_building_block_diversity, store_nodes=store_nodes, verbose=
        verbose, replicate=replicate)
    print('Generating molecules...')
    start_time = datetime.now()
    nodes = generator.generate(n_rollout=n_rollout)
    stats = {'mcts_time': datetime.now() - start_time,
        'num_nonzero_reaction_molecules': len(nodes),
        'approx_num_nodes_searched': generator.approx_num_nodes_searched}
    print(f"MCTS time = {stats['mcts_time']}")
    print(
        f"Number of full molecule, nonzero reaction nodes = {stats['num_nonzero_reaction_molecules']:,}"
        )
    print(
        f"Approximate total number of nodes searched = {stats['approx_num_nodes_searched']:,}"
        )
    if store_nodes:
        stats['num_nodes_searched'] = generator.num_nodes_searched
        print(
            f"Total number of nodes searched = {stats['num_nodes_searched']:,}"
            )
    pd.DataFrame(data=[stats]).to_csv(save_dir / 'mcts_stats.csv', index=False)
    print('Saving molecules...')
    save_generated_molecules(nodes=nodes, building_block_id_to_smiles=
        building_block_id_to_smiles, save_path=save_dir / 'molecules.csv')


def evaluate(cfg: DictConfig):
    if cfg.data_label in ["esol", "freesolv", "lipop"]:
        # task_type = "reg"
        reg = True
        metric = "rmse"
    else:
        # task_type = "cla"
        reg = False
        metric = "roc_auc"

    # set dataloader config
    eval_dataloader_cfg = {
        "dataset": {
            "name": "IFMMoeDataset",
            "input_keys": ("x",),
            "label_keys": (
                "y",
                "mask",
            ),
            "data_dir": cfg.data_dir,
            "data_mode": "train",
            "data_label": cfg.data_label,
        },
        "batch_size": 128,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "num_workers": 1,
    }

    # set constraint
    sup_constraint = ppsci.constraint.SupervisedConstraint(
        eval_dataloader_cfg,
        output_expr={"pred": lambda out: out["pred"]},
        loss=ppsci.loss.FunctionalLoss(get_train_loss_func(reg)),
        name="Sup",
    )

    inputs = sup_constraint.data_loader.dataset.data_tr_x.shape[1]
    tasks = sup_constraint.data_loader.dataset.task_dict[cfg.data_label]

    hyper_paras = cfg.HYPER_OPT[cfg.data_label]
    hidden_units = [
        hyper_paras["hidden_unit1"],
        hyper_paras["hidden_unit2"],
        hyper_paras["hidden_unit3"],
    ]
    print(f"hyper_params = {hyper_paras}")

    # set model
    model = ppsci.arch.IFMMLP(
        # **cfg.MODEL,
        input_keys=("x",),
        output_keys=("pred",),
        hidden_units=hidden_units,
        embed_name=cfg.MODEL.embed_name,
        inputs=inputs,
        outputs=len(tasks),
        d_out=hyper_paras["d_out"],
        sigma=hyper_paras["sigma"],
        dp_ratio=hyper_paras["dropout"],
        reg=reg,
        first_omega_0=hyper_paras["omega0"],
        hidden_omega_0=hyper_paras["omega1"],
    )

    # set validator
    eval_dataloader_cfg = {
        "dataset": {
            "name": "IFMMoeDataset",
            "input_keys": ("x",),
            "label_keys": (
                "y",
                "mask",
            ),
            "data_dir": cfg.data_dir,
            "data_mode": "test",
            "data_label": cfg.data_label,
        },
        "batch_size": cfg.EVAL.batch_size,
        "sampler": {
            "name": "BatchSampler",
            "drop_last": False,
            "shuffle": True,
        },
        "num_workers": 1,
    }

    rmse_validator = ppsci.validate.SupervisedValidator(
        eval_dataloader_cfg,
        loss=ppsci.loss.FunctionalLoss(get_train_loss_func(reg)),
        output_expr={"pred": lambda out: out["pred"]},
        metric={
            "MyMeter": ppsci.metric.FunctionalMetric(get_val_loss_func(reg, metric))
        },
        name="MyMeter_validator",
    )
    if not reg:
        pos_weights = rmse_validator.data_loader.dataset.pos_weights
        rmse_validator.loss = ppsci.loss.FunctionalLoss(
            get_train_loss_func(reg, pos_weights)
        )

    validator = {rmse_validator.name: rmse_validator}

    if cfg.EVAL.pretrained_model_path:
        pretrained_model_path = cfg.EVAL.pretrained_model_path
    else:
        t_epoch = cfg.HYPER_OPT[cfg.data_label].epoch
        load_epoch = t_epoch - t_epoch % cfg.TRAIN.save_freq
        pretrained_model_path = os.path.join(
            cfg.output_dir, "checkpoints", "epoch_" + str(load_epoch) + ".pdparams"
        )

    solver = ppsci.solver.Solver(
        model,
        output_dir=cfg.output_dir,
        log_freq=cfg.log_freq,
        validator=validator,
        pretrained_model_path=pretrained_model_path,
        eval_with_no_grad=cfg.EVAL.eval_with_no_grad,
    )

    # evaluate model
    solver.eval()


@hydra.main(version_base=None, config_path="./conf", config_name="chemprop.yaml")
def main(cfg: DictConfig):
    if cfg.mode == "train":
        train(cfg)
    elif cfg.mode == "eval":
        evaluate(cfg)
    elif cfg.mode == 'pre-compute':
        pre_compute()
    elif cfg.mode == 'generate':
        generate()
    else:
        raise ValueError(f"cfg.mode should in ['train', 'eval'], but got '{cfg.mode}'")


if __name__ == "__main__":
    main()

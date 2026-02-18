import warnings

import torch

from monai.data import MetaTensor
from monai.engines.utils import IterationEvents
from monai.utils.enums import CommonKeys as Keys
import torch


import torch
from typing import Dict, Any

from monai.engines import SupervisedTrainer


compile = False


def iteration(engine: SupervisedTrainer, batchdata: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Callback function for the Supervised Training processing logic of 1 iteration in Ignite Engine.
    Return below items in a dictionary:
        - IMAGE: image Tensor data for model input, already moved to device.
        - LABEL: label Tensor data corresponding to the image, already moved to device.
        - PRED: prediction result of model.
        - LOSS: loss value computed by loss function.

    Args:
        engine: `SupervisedTrainer` to execute operation for an iteration.
        batchdata: input data for this iteration, usually can be dictionary or tuple of Tensor data.

    Raises:
        ValueError: When ``batchdata`` is None.

    """
    if batchdata is None:
        raise ValueError("Must provide batch data for current iteration.")
    batch = engine.prepare_batch(batchdata, engine.state.device, engine.non_blocking, **engine.to_kwargs)
    if len(batch) == 2:
        inputs, targets = batch
        args: tuple = ()
        kwargs: dict = {}
    elif len(batch) == 3:
        inputs, inputs_2, targets = batch
        args: tuple = ()
        kwargs: dict = {}
    else:
        inputs, targets, args, kwargs = batch
    # FIXME: workaround for https://github.com/pytorch/pytorch/issues/117026
    if compile:
        inputs_meta, targets_meta, inputs_applied_operations, targets_applied_operations = None, None, None, None
        if isinstance(inputs, MetaTensor):
            warnings.warn(
                "Will convert to PyTorch Tensor if using compile, and casting back to MetaTensor after the forward pass."
            )
            inputs, inputs_meta, inputs_applied_operations = (
                inputs.as_tensor(),
                inputs.meta,
                inputs.applied_operations,
            )
        if isinstance(targets, MetaTensor):
            targets, targets_meta, targets_applied_operations = (
                targets.as_tensor(),
                targets.meta,
                targets.applied_operations,
            )

    # put iteration outputs into engine.state
    engine.state.output = {Keys.IMAGE: inputs, Keys.LABEL: targets}

    def _compute_pred_loss():
        outputs_v1 = engine.inferer(inputs, engine.network, *args, **kwargs)
        outputs_v2 =  engine.inferer(inputs_2, engine.network, *args, **kwargs)
        
        flat_out_v1 = outputs_v1.flatten(start_dim=1, end_dim=-1)
        flat_out_v2 = outputs_v2.flatten(start_dim=1, end_dim=-1)
        
        engine.state.output[Keys.PRED] = {"outputs_v1": outputs_v1, "flat_out_v1": flat_out_v1, "flat_out_v2":flat_out_v2}
        engine.fire_event(IterationEvents.FORWARD_COMPLETED)
        loss = engine.loss_function({"outputs_v1": outputs_v1, "flat_out_v1": flat_out_v1, "flat_out_v2":flat_out_v2}, targets)
        engine.state.output[Keys.LOSS] = loss["total_loss"].mean()
        engine.fire_event(IterationEvents.LOSS_COMPLETED)

    engine.network.train()
    engine.optimizer.zero_grad(set_to_none=engine.optim_set_to_none)

    if engine.amp and engine.scaler is not None:
        with torch.cuda.amp.autocast(**engine.amp_kwargs):
            _compute_pred_loss()
        engine.scaler.scale(engine.state.output[Keys.LOSS]).backward()
        engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
        engine.scaler.step(engine.optimizer)
        engine.scaler.update()
    else:
        _compute_pred_loss()
        engine.state.output[Keys.LOSS].backward()
        engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
        engine.optimizer.step()
    # copy back meta info
    if compile:
        if inputs_meta is not None:
            engine.state.output[Keys.IMAGE] = MetaTensor(
                inputs, meta=inputs_meta, applied_operations=inputs_applied_operations
            )
            engine.state.output[Keys.PRED] = MetaTensor(
                engine.state.output[Keys.PRED], meta=inputs_meta, applied_operations=inputs_applied_operations
            )
        if targets_meta is not None:
            engine.state.output[Keys.LABEL] = MetaTensor(
                targets, meta=targets_meta, applied_operations=targets_applied_operations
            )
    engine.fire_event(IterationEvents.MODEL_COMPLETED)

    return engine.state.output


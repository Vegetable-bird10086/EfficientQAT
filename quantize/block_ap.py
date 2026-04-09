import torch
import torch.nn as nn
import torch.nn.functional as F
import quantize.int_linear_fake as int_linear_fake
import quantize.int_linear_real as int_linear_real
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy
import math
import utils
import pdb
import gc
from quantize.utils import (
    quant_parameters,weight_parameters,trainable_parameters,
    set_quant_state,quant_inplace,set_quant_parameters,
    set_weight_parameters,trainable_parameters_num,get_named_linears,set_op_by_name)
from quantize.config import EfficientQATQuantConfig, QuantizationSpec
import time
from datautils_block import BlockTrainDataset
from torch.utils.data import DataLoader
import shutil
import os
import json


def _module_grad_statistics(module: nn.Module) -> dict | None:
    total_sq = 0.0
    total_abs = 0.0
    total_elems = 0
    max_abs = 0.0
    grad_tensors = 0
    trainable_param_names = []

    for name, param in module.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        grad = param.grad.detach().float()
        total_sq += float(grad.pow(2).sum().item())
        total_abs += float(grad.abs().sum().item())
        total_elems += grad.numel()
        max_abs = max(max_abs, float(grad.abs().max().item()))
        grad_tensors += 1
        trainable_param_names.append(name)

    if total_elems == 0:
        return None

    return {
        "l2_norm": math.sqrt(total_sq),
        "mean_abs_grad": total_abs / total_elems,
        "max_abs_grad": max_abs,
        "grad_tensors": grad_tensors,
        "trainable_param_names": trainable_param_names,
    }


class GradientSensitivityTracker:
    def __init__(self, sort_by: str = "avg_mean_abs_grad"):
        self.sort_by = sort_by
        self._stats: dict[str, dict] = {}

    def update(self, block_index: int, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if not isinstance(module, int_linear_fake.QuantLinear):
                continue
            grad_stats = _module_grad_statistics(module)
            if grad_stats is None:
                continue
            module_name = f"layers.{block_index}.{name}"
            entry = self._stats.setdefault(
                module_name,
                {
                    "module_name": module_name,
                    "updates": 0,
                    "sum_l2_norm": 0.0,
                    "sum_mean_abs_grad": 0.0,
                    "max_abs_grad": 0.0,
                    "grad_tensors": grad_stats["grad_tensors"],
                    "trainable_param_names": grad_stats["trainable_param_names"],
                },
            )
            entry["updates"] += 1
            entry["sum_l2_norm"] += grad_stats["l2_norm"]
            entry["sum_mean_abs_grad"] += grad_stats["mean_abs_grad"]
            entry["max_abs_grad"] = max(entry["max_abs_grad"], grad_stats["max_abs_grad"])

    def ranked(self) -> list[dict]:
        ranked = []
        for entry in self._stats.values():
            updates = max(1, entry["updates"])
            ranked.append(
                {
                    "module_name": entry["module_name"],
                    "updates": entry["updates"],
                    "avg_l2_norm": entry["sum_l2_norm"] / updates,
                    "avg_mean_abs_grad": entry["sum_mean_abs_grad"] / updates,
                    "max_abs_grad": entry["max_abs_grad"],
                    "grad_tensors": entry["grad_tensors"],
                    "trainable_param_names": entry["trainable_param_names"],
                }
            )
        ranked.sort(key=lambda item: item[self.sort_by], reverse=True)
        return ranked

    def save(self, output_dir: str) -> str:
        report_path = os.path.join(output_dir, "gradient_sensitivity_ranking.json")
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(self.ranked(), handle, indent=2)
        return report_path


def _layer_hidden_states(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _layer_forward(
    layer,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    cache_position=None,
    rotary_emb=None,
    position_embeddings=None,
):
    forward_kwargs = {}
    if attention_mask is not None:
        forward_kwargs["attention_mask"] = attention_mask
    if position_ids is not None:
        forward_kwargs["position_ids"] = position_ids
    if cache_position is not None:
        forward_kwargs["cache_position"] = cache_position
    if position_embeddings is not None:
        forward_kwargs["position_embeddings"] = position_embeddings
    elif rotary_emb is not None and position_ids is not None:
        forward_kwargs["position_embeddings"] = rotary_emb(hidden_states, position_ids.to(hidden_states.device))
    return _layer_hidden_states(layer(hidden_states, **forward_kwargs))


def update_dataset(
    layer,
    dataset,
    dev,
    attention_mask,
    position_ids,
    cache_position=None,
    rotary_emb=None,
    position_embeddings=None,
):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for index, inps in enumerate(dataset):
                inps = inps.to(dev)
                if len(inps.shape)==2:
                    inps = inps.unsqueeze(0)
                new_data = _layer_forward(
                    layer,
                    inps,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    rotary_emb=rotary_emb,
                    position_embeddings=position_embeddings,
                ).to('cpu')
                dataset.update_data(index,new_data)

                    
def block_ap(
    model,
    args,
    trainloader,
    valloader,
    logger=None,
    quant_config: EfficientQATQuantConfig | None = None,
):
    logger.info("Starting ...")
    if args.off_load_to_disk:
        logger.info("offload the training dataset to disk, saving CPU memory, but may slowdown the training due to additional I/O...")
    grad_tracker = GradientSensitivityTracker(sort_by=args.grad_sensitivity_sort_by) if getattr(args, "log_grad_sensitivity", False) else None
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    # step 1: move embedding layer and first layer to target device, only suppress llama models now
    layers = model.model.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = torch.float16

    # step 2: init dataset
    flag = time.time()
    if args.off_load_to_disk:
        fp_train_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_train'
        fp_val_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_val'
        quant_train_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_train'
        quant_val_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_val'
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
    else:
        fp_train_cache_path = None
        fp_val_cache_path = None
        quant_train_cache_path = None
        quant_val_cache_path = None
    fp_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                model.config.hidden_size, args.batch_size, dtype, cache_path=fp_train_cache_path,off_load_to_disk=args.off_load_to_disk)
    fp_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                model.config.hidden_size, args.batch_size, dtype, cache_path=fp_val_cache_path,off_load_to_disk=args.off_load_to_disk)
    
    # step 3: catch the input of thefirst layer 
    class Catcher(nn.Module):
        def __init__(self, module, dataset):
            super().__init__()
            self.module = module
            self.dataset = dataset
            self.index = 0
            self.attention_mask = None
            self.position_ids = None
            self.cache_position = None
            self.position_embeddings = None

        def forward(self, inp, **kwargs):
            self.dataset.update_data(self.index, inp.squeeze(0).to('cpu'))
            self.index += 1
            if self.attention_mask is None:
                self.attention_mask = kwargs.get("attention_mask")
            if self.position_ids is None:
                self.position_ids = kwargs.get("position_ids")
            if self.cache_position is None:
                self.cache_position = kwargs.get("cache_position")
            if self.position_embeddings is None:
                self.position_embeddings = kwargs.get("position_embeddings")
            raise ValueError
    
    # step 3.1: catch the input of training set
    layers[0] = Catcher(layers[0],fp_train_inps)
    iters = len(trainloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([trainloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    layers[0] = layers[0].module

    # step 3.2: catch the input of validation set
    layers[0] = Catcher(layers[0],fp_val_inps)
    iters = len(valloader)//args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([valloader[j][0] for j in range(i*args.batch_size,(i+1)*args.batch_size)],dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                pass
    attention_mask = layers[0].attention_mask
    position_ids = layers[0].position_ids
    cache_position = layers[0].cache_position
    position_embeddings = layers[0].position_embeddings
    layers[0] = layers[0].module
    if attention_mask is not None:
        if attention_mask.shape[0] == 1 and args.batch_size > 1:
            attention_mask_batch = attention_mask.repeat(args.batch_size,1,1,1).float()
        else:
            attention_mask_batch = attention_mask.float()
    else:
        logger.info(
            "No attention mask caught from the first layer."
            " Seems that model's attention works without a mask."
        )
        attention_mask_batch = None
    
    # step 4: move embedding layer and first layer to cpu
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, 'rotary_emb'):
        # for llama-3.1
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    # step 5: copy fp input as the quant input, they are same at the first layer
    if args.off_load_to_disk:
        # copy quant input from fp input, they are same in first layer
        shutil.copytree(fp_train_cache_path, quant_train_cache_path)
        shutil.copytree(fp_val_cache_path, quant_val_cache_path)
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
    else:
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_train_cache_path,off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen, 
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=quant_val_cache_path,off_load_to_disk=args.off_load_to_disk)
        for index,data in enumerate(fp_train_inps):
            quant_train_inps.update_data(index, data)
        for index,data in enumerate(fp_val_inps):
            quant_val_inps.update_data(index, data)

    # step 6: start training    
    loss_func = torch.nn.MSELoss()
    for block_index in range(len(layers)):
        logger.info(f"=== Start quantize blocks {block_index}===")
        # step 6.1: replace torch.nn.Linear with QuantLinear for QAT
        layer = layers[block_index].to(dev)
        block_rotary_emb = getattr(model.model, "rotary_emb", None)
        if block_rotary_emb is not None:
            block_rotary_emb = block_rotary_emb.to(dev)
        qlayer = copy.deepcopy(layer)
        for name, module in qlayer.named_modules():
            if isinstance(module,torch.nn.Linear):
                resolved_name = f"model.layers.{block_index}.{name}"
                spec = quant_config.resolve(resolved_name, module.in_features) if quant_config is not None else QuantizationSpec(bits=args.wbits, group_size=args.group_size)
                if not spec.should_quantize:
                    continue
                quantlinear = int_linear_fake.QuantLinear(module, args.wbits, args.group_size, quant_spec=spec)
                set_op_by_name(qlayer, name, quantlinear)
                del module  
        qlayer.to(dev)
        
        
        # step 6.2: obtain output of full-precision model for MSE
        set_quant_state(qlayer,weight_quant=False) # deactivate quantization for obtaining ground truth
        if args.epochs > 0:
            update_dataset(qlayer,fp_train_inps,dev,attention_mask,position_ids,cache_position=cache_position,rotary_emb=block_rotary_emb,position_embeddings=position_embeddings)
            update_dataset(qlayer,fp_val_inps,dev,attention_mask,position_ids,cache_position=cache_position,rotary_emb=block_rotary_emb,position_embeddings=position_embeddings)
        set_quant_state(qlayer,weight_quant=True)  # activate quantization
        
        
        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      # fp32 is required for AMP training
            # step 6.3: create optimizer and learning rate schedule
            param = []
            assert args.quant_lr > 0 or args.weight_lr > 0
            param_group_index = 0
            total_training_iteration = args.epochs * args.train_size / args.batch_size 
            if args.quant_lr > 0:
                set_quant_parameters(qlayer,True)
                param.append({"params":quant_parameters(qlayer),"lr":args.quant_lr})
                empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)], lr=args.quant_lr)
                quant_scheduler = CosineAnnealingLR(empty_optimizer_1, T_max=total_training_iteration, eta_min=args.quant_lr/args.min_lr_factor)
                quant_index = param_group_index
                param_group_index += 1
            else:
                set_quant_parameters(qlayer,False)
                
            if args.weight_lr > 0:
                set_weight_parameters(qlayer,True)
                param.append({"params":weight_parameters(qlayer),"lr":args.weight_lr})
                empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)], lr=args.weight_lr)
                weight_scheduler = CosineAnnealingLR(empty_optimizer_2, T_max=total_training_iteration, eta_min=args.weight_lr/args.min_lr_factor)
                weight_index = param_group_index
                param_group_index += 1
            else:
                set_weight_parameters(qlayer,False)
            optimizer = torch.optim.AdamW(param, weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            trainable_number = trainable_parameters_num(qlayer)
            print(f"trainable parameter number: {trainable_number/1e6}M")

            best_val_loss = 1e6
            early_stop_flag = 0
            for epoch in range(args.epochs):
                # step: 6.4 training
                loss_list = []
                norm_list = []
                start_time = time.time()
                for index, (quant_inps, fp_inps) in enumerate(zip(quant_train_inps, fp_train_inps)):    
                    # obtain output of quantization model
                    with torch.cuda.amp.autocast():
                        input = quant_inps.to(dev)
                        label = fp_inps.to(dev)
                        quant_out = _layer_forward(
                            qlayer,
                            input,
                            attention_mask=attention_mask_batch,
                            position_ids=position_ids,
                            cache_position=cache_position,
                            rotary_emb=block_rotary_emb,
                            position_embeddings=position_embeddings,
                        )
                        reconstruction_loss = loss_func(label, quant_out)
                        loss =  reconstruction_loss

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                    loss_list.append(reconstruction_loss.detach().cpu())
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters=trainable_parameters(qlayer)).cpu()
                    if grad_tracker is not None:
                        grad_tracker.update(block_index, qlayer)
                    norm_list.append(norm.data)

                    # adjust lr
                    if args.quant_lr > 0:
                        quant_scheduler.step()
                        optimizer.param_groups[quant_index]['lr'] = quant_scheduler.get_lr()[0]
                    if args.weight_lr >0 :
                        weight_scheduler.step()
                        optimizer.param_groups[weight_index]['lr'] = weight_scheduler.get_lr()[0]

                # step 6.5: calculate validation loss
                val_loss_list = []
                for index, (quant_inps,fp_inps) in enumerate(zip(quant_val_inps, fp_val_inps)):  
                    # obtain output of quantization model
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            input = quant_inps.to(dev)
                            label = fp_inps.to(dev)
                            quant_out = _layer_forward(
                                qlayer,
                                input,
                                attention_mask=attention_mask_batch,
                                position_ids=position_ids,
                                cache_position=cache_position,
                                rotary_emb=block_rotary_emb,
                                position_embeddings=position_embeddings,
                            )
                            reconstruction_loss = loss_func(label, quant_out)
                    val_loss_list.append(reconstruction_loss.cpu())
                 
                train_mean_num = min(len(loss_list),64) # calculate the average training loss of last train_mean_num samples
                loss_mean = torch.stack(loss_list)[-(train_mean_num-1):].mean()
                val_loss_mean = torch.stack(val_loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"blocks {block_index} epoch {epoch} recon_loss:{loss_mean} val_loss:{val_loss_mean} quant_lr:{quant_scheduler.get_lr()[0]} norm:{norm_mean:.8f} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024**2} time {time.time()-start_time} ")
                if val_loss_mean < best_val_loss:
                    best_val_loss = val_loss_mean
                else:
                    early_stop_flag += 1
                    if args.early_stop > 0 and early_stop_flag >=args.early_stop:
                        break
            optimizer.zero_grad()
            del optimizer

        # step 6.6: directly replace the weight with fake quantization
        qlayer.half()
        quant_inplace(qlayer)
        set_quant_state(qlayer,weight_quant=False)  # weight has been quantized inplace

        # step 6.7: update inputs of quantization model
        if args.epochs>0:
            update_dataset(qlayer,quant_train_inps,dev,attention_mask,position_ids,cache_position=cache_position,rotary_emb=block_rotary_emb,position_embeddings=position_embeddings)
            update_dataset(qlayer,quant_val_inps,dev,attention_mask,position_ids,cache_position=cache_position,rotary_emb=block_rotary_emb,position_embeddings=position_embeddings)
        layers[block_index] = qlayer.to("cpu")
        if block_rotary_emb is not None:
            model.model.rotary_emb = block_rotary_emb.cpu()

        # step 7: pack quantized weights into low-bits format, note that this process is slow on poor CPU or busy CPU
        if args.real_quant:
            named_linears = get_named_linears(qlayer, int_linear_fake.QuantLinear)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scale.clamp(1e-4,1e4).detach()
                zeros = module.weight_quantizer.zero_point.detach().cuda().round().cpu()
                group_size = module.weight_quantizer.group_size
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1).transpose(0,1).contiguous()
                zeros = zeros.view(dim0,-1).transpose(0,1).contiguous()
                q_linear = int_linear_real.QuantLinear(
                    module.quant_spec.bits,
                    group_size,
                    module.in_features,
                    module.out_features,
                    not module.bias is None,
                    mapping=module.quant_spec.mapping,
                    train_scale=module.quant_spec.train_scale,
                    train_zero_point=module.quant_spec.train_zero_point,
                    quant_spec=module.quant_spec,
                )
                q_linear.pack(module.cpu(),  scales.float().cpu(), zeros.float().cpu())
                set_op_by_name(qlayer, name, q_linear)       
                logger.info(f"pack quantized {name} finished")
                del module        
        del layer
        torch.cuda.empty_cache()

    if grad_tracker is not None and args.output_dir:
        ranked = grad_tracker.ranked()
        logger.info(
            "=== Gradient sensitivity ranking (sorted by %s, top %d) ===",
            args.grad_sensitivity_sort_by,
            min(args.grad_sensitivity_topk, len(ranked)),
        )
        for index, item in enumerate(ranked[: args.grad_sensitivity_topk], start=1):
            logger.info(
                "[%d] %s avg_mean_abs_grad=%.8e avg_l2_norm=%.8e max_abs_grad=%.8e updates=%d",
                index,
                item["module_name"],
                item["avg_mean_abs_grad"],
                item["avg_l2_norm"],
                item["max_abs_grad"],
                item["updates"],
            )
        report_path = grad_tracker.save(args.output_dir)
        logger.info("saved gradient sensitivity report to %s", report_path)

    # delete cached dataset
    if args.off_load_to_disk:
        for path in [fp_train_cache_path,fp_val_cache_path,quant_train_cache_path,quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)

    torch.cuda.empty_cache()
    gc.collect()                    
    model.config.use_cache = use_cache
    return model

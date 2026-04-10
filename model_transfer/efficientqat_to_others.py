import torch
from datautils_block import test_ppl
from gptqmodel import GPTQModel, QuantizeConfig, get_backend
from pathlib import Path
import time
from hf_compat import load_auto_tokenizer, resolve_hf_token
from quantize.config import load_quant_config, maybe_load_quant_config, is_uniform_quant_config

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, type=str, help="direction for saving quantization model")
    parser.add_argument("--wbits", type=int, default=4, help="quantization bits")
    parser.add_argument("--group_size", type=int, default=128, help="quantization group size")
    parser.add_argument("--target_format", default='gptq', type=str, help="target checkpoint format")
    parser.add_argument("--quant_config", default=None, type=str, help="optional JSON quantization config; defaults to the metadata in --model")
    parser.add_argument("--trust_remote_code", action="store_true", help="enable trust_remote_code when loading tokenizer/model")
    parser.add_argument("--token", default=None, type=str, help="HF token for gated/private models")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--test_speed", action="store_true")
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving quantization model")

    


    args = parser.parse_args()
    hf_token = resolve_hf_token(token=args.token)
    if args.quant_config is not None:
        config_path = Path(args.quant_config)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Explicit quantization config not found: {config_path}. "
                "If you intended to pass an absolute path, make sure it starts with '/'."
            )
        quant_config = load_quant_config(str(config_path), default_bits=args.wbits, default_group_size=args.group_size)
    else:
        quant_config = maybe_load_quant_config(args.model, default_bits=args.wbits, default_group_size=args.group_size)
    if not is_uniform_quant_config(quant_config):
        raise NotImplementedError(
            "GPTQ/BitBLAS export currently expects a uniform quantization config. "
            "Per-layer overrides should use the TorchAO/ExecuTorch metadata path instead."
        )
    tokenizer = load_auto_tokenizer(
        args.model,
        use_fast=False,
        legacy=False,
        trust_remote_code=args.trust_remote_code,
        token=hf_token,
    )
    quant_config = QuantizeConfig(
    bits=args.wbits,  
    group_size=args.group_size,
    sym=False,
    desc_act=False,
    format='gptq_v2',
    )
    if args.target_format == 'gptq':
        # EXLLAMA_V2 is faster in 4-bit, and can inference correctly. However, it has some bug in saving models.
        # Therefore, we choose triton backend as default. Note that the saving model can also be loaded by exllama too.
        model = GPTQModel.from_quantized(
            args.model,
            device_map='auto',
            torch_dtype=torch.float16,
            quantize_config=quant_config,
            backend=get_backend('TRITON'),
            trust_remote_code=args.trust_remote_code,
            use_auth_token=hf_token,
        )

    elif args.target_format == 'bitblas':
        # take a lone time for the first time runing
        try:
            model = GPTQModel.from_quantized(
                args.model,
                device_map='auto',
                torch_dtype=torch.float16,
                quantize_config=quant_config,
                backend=get_backend('BITBLAS'),
                trust_remote_code=args.trust_remote_code,
                use_auth_token=hf_token,
            )
            args.eval_ppl = False # BitBLAS have bug, which should re-load model for evaluation otherwise would cause wrong outputs
        except:
            model = GPTQModel.from_quantized(
                args.model,
                device_map='auto',
                torch_dtype=torch.float16,
                backend=get_backend('BITBLAS'),
                trust_remote_code=args.trust_remote_code,
                use_auth_token=hf_token,
            )
    else:
        raise NotImplementedError

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        print("start saving model")
        model.quantize_config.model_file_base_name=None # trick to avoid one saving bug in GPTQModel
        model.save_quantized(args.save_dir,max_shard_size='8GB')  
        tokenizer.save_pretrained(args.save_dir) 
        quant_config.save(args.save_dir)
        print(f"save model to {args.save_dir} success")

    model.model.cuda()
    
    if args.eval_ppl:
        datasets = ["wikitext2"]
        ppl_results = test_ppl(model, tokenizer, datasets, 2048)
        for dataset in ppl_results:
            print(f'{dataset} perplexity after transfering: {ppl_results[dataset]:.2f}')
    if args.test_speed:
        prompt = "Write a poem about large language model:"
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
        start_time = time.time()
        output = model.generate(inputs=input_ids, do_sample=True, top_k=10, max_new_tokens=256)
        end_time = time.time()
        speed = len(output[0])/(end_time-start_time)
        print(tokenizer.decode(output[0]))
        print(f"generation speed:{speed:.1f}token/s")
        

if __name__ =='__main__':
    main()

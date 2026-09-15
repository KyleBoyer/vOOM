#!/usr/bin/env python3
"""Fresh-process native grammar equivalence probe; no MLX or model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backend',choices=('stock','cpu'),required=True)
    parser.add_argument('--result',required=True)
    args=parser.parse_args()
    started=time.perf_counter()
    from tokenizers import Tokenizer
    model=ROOT/'models/Huihui-Qwen3.8-27B-abliterated-mlx-all-mxfp4'
    tokenizer=Tokenizer.from_file(str(model/'tokenizer.json'))
    raw=json.loads((model/'config.json').read_text());raw=raw.get('text_config',raw)
    stops=raw['eos_token_id'];stops=[stops] if isinstance(stops,int) else stops
    if args.backend=='cpu':
        from runtime.xgrammar_cpu import backend
        xgr=backend()
        info=xgr.TokenizerInfo.from_engine(NS(tokenizer=tokenizer,cfg=NS(vocab_size=raw['vocab_size'],eos_token_ids=stops)))
    else:
        import xgrammar as xgr
        from transformers import AutoTokenizer
        hf=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
        info=xgr.TokenizerInfo.from_huggingface(hf,vocab_size=raw['vocab_size'],stop_token_ids=stops)
        del hf
    compiler=xgr.GrammarCompiler(info,max_threads=4,cache_enabled=True)
    digest=hashlib.sha256()
    vocab=info.decoded_vocab
    for value in vocab:
        digest.update(len(value).to_bytes(8,'little'));digest.update(value)
    schemas=[
        ({'type':'object','properties':{'city':{'type':'string'}},'required':['city'],'additionalProperties':False},'{"city": "東京"}'),
        ({'type':'object','properties':{'count':{'type':'integer'},'ok':{'type':'boolean'}},'required':['count','ok'],'additionalProperties':False},'{"count": 123, "ok": true}'),
        ({'type':'array','items':{'type':'string','enum':['alpha','beta','gamma']}},'["alpha", "beta", "gamma"]'),
        ({'type':'object','properties':{'files':{'type':'array','items':{'type':'string'}}},'required':['files'],'additionalProperties':False},'{"files": ["a.py", "b.txt", "資料.md"]}'),
    ]
    cases=[]
    for schema,text in schemas:
        for mode in ('json','tool'):
            if mode=='json':
                compiled=compiler.compile_json_schema(schema,any_whitespace=False,strict_mode=True)
                output=text;terminate=True
            else:
                grammar=xgr.Grammar.from_structural_tag([
                    xgr.StructuralTagItem(begin='<tool_call>',schema=schema,end='</tool_call>')],['<tool_call>'])
                compiled=compiler.compile_grammar(grammar)
                output='Checking. <tool_call>'+text+'</tool_call>';terminate=False
            matcher=xgr.GrammarMatcher(compiled,terminate_without_stop_token=terminate)
            mask=xgr.allocate_token_bitmask(1,raw['vocab_size'])
            def mask_hash(m):
                if m.is_terminated():return 'terminated'
                needed=m.fill_next_token_bitmask(mask)
                return [bool(needed),hashlib.sha256(mask.numpy().tobytes()).hexdigest()]
            tokens=tokenizer.encode(output,add_special_tokens=False).ids
            trajectory=[]
            for token in tokens:
                before=mask_hash(matcher)
                fork=matcher.fork()
                assert mask_hash(fork)==before
                accepted=bool(matcher.accept_token(token))
                assert accepted and fork.accept_token(token),(mode,output,token)
                after=mask_hash(matcher)
                assert mask_hash(fork)==after
                matcher.rollback(1)
                assert mask_hash(matcher)==before
                assert matcher.accept_token(token)
                assert mask_hash(matcher)==after
                trajectory.append([token,before,after])
            cases.append({'mode':mode,'trajectory':trajectory,'completed':bool(matcher.is_completed()),'terminated':bool(matcher.is_terminated())})
    result={'passed':True,'backend':args.backend,'vocab_count':len(vocab),
        'vocab_sha256':digest.hexdigest(),'metadata':json.loads(info.dump_metadata()),
        'special_token_ids':info.special_token_ids,'cases':cases,
        'wall_s':time.perf_counter()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'torch_loaded':'torch' in sys.modules,'transformers_loaded':'transformers' in sys.modules}
    if args.backend=='cpu':assert not result['torch_loaded'] and not result['transformers_loaded']
    path=Path(args.result)
    with path.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))


if __name__=='__main__':main()

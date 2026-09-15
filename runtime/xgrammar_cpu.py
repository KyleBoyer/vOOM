"""Opt-in NumPy/DLPack facade over the installed, pinned XGrammar C++ backend.

Reuse upstream grammar/compiler modules under a private package namespace.
Avoid their unrelated PyTorch matcher and Transformers tokenizer wrappers.
No grammar, model-logit or token acceptance algorithm is replaced here.
"""
from __future__ import annotations

import _imp
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np

_API = None
_LOCK = threading.Lock()
_PACKAGE = 'runtime._xgrammar_cpu_upstream'


def backend():
    global _API
    with _LOCK:
        if _API is not None:
            return _API
        if importlib.metadata.version('xgrammar') != '0.2.3' or importlib.metadata.version('apache-tvm-ffi') != '0.1.12':
            raise RuntimeError('CPU XGrammar facade requires qualified xgrammar0.2.3 / tvm-ffi0.1.12')
        if 'xgrammar' in sys.modules or 'torch' in sys.modules:
            raise RuntimeError('CPU XGrammar facade requires a fresh process without PyTorch/XGrammar')
        root=Path(importlib.metadata.distribution('xgrammar').locate_file('xgrammar'))
        # TVM explicitly supports missing optional torch. Serialize imports and
        # expose that supported ImportError path only while initializing TVM;
        # restore the import table even on failure. No fake torch implementation.
        _imp.acquire_lock()
        try:
            if 'torch' in sys.modules:
                raise RuntimeError('PyTorch was loaded concurrently')
            sys.modules['torch']=None
            try:
                from tvm_ffi.libinfo import load_lib_module
                native=load_lib_module('xgrammar','xgrammar_bindings')
            finally:
                sys.modules.pop('torch',None)
            package=ModuleType(_PACKAGE);package.__path__=[str(root)]
            sys.modules[_PACKAGE]=package
            base=importlib.import_module(_PACKAGE+'.base')
            core=base._core

            class TokenizerInfo(base.XGRObject):
                @classmethod
                def from_engine(cls,engine):
                    tokenizer=engine.tokenizer
                    vocab=tokenizer.get_vocab()
                    size=int(engine.cfg.vocab_size)
                    encoded=['']*size
                    for token,index in vocab.items():
                        if type(index) is not int or index<0:
                            raise ValueError('invalid tokenizer index')
                        if index<size:encoded[index]=token
                    metadata=json.loads(core.TokenizerInfo._detect_metadata_from_hf(tokenizer.to_str()))
                    result=cls.__new__(cls)
                    result._init_handle(core.TokenizerInfo(encoded,int(metadata['vocab_type']),size,
                        list(engine.cfg.eos_token_ids),bool(metadata['add_prefix_space'])))
                    return result
                def dump_metadata(self):return str(self._handle.dump_metadata())
                @property
                def decoded_vocab(self):return list(self._handle.decoded_vocab())
                @property
                def special_token_ids(self):return list(self._handle.special_token_ids())

            token_module=ModuleType(_PACKAGE+'.tokenizer_info')
            token_module.TokenizerInfo=TokenizerInfo
            sys.modules[token_module.__name__]=token_module
            grammar=importlib.import_module(_PACKAGE+'.grammar')
            compiler=importlib.import_module(_PACKAGE+'.compiler')
        finally:
            _imp.release_lock()

        class Bitmask(np.ndarray):
            def numpy(self):return np.asarray(self)

        def allocate(batch,size):
            return np.full((batch,(size+31)//32),-1,dtype=np.int32).view(Bitmask)

        class GrammarMatcher:
            def __init__(self,compiled,*,terminate_without_stop_token=False):
                self._handle=core.GrammarMatcher(compiled._handle,None,terminate_without_stop_token,-1)
            def fill_next_token_bitmask(self,mask):
                return self._handle.fill_next_token_bitmask(np.asarray(mask),0,False)
            def accept_token(self,token):return self._handle.accept_token(int(token),False)
            def accept_string(self,text):return self._handle.accept_string(text,False)
            def is_completed(self):return self._handle.is_completed()
            def is_terminated(self):return self._handle.is_terminated()
            def find_jump_forward_string(self):return str(self._handle.find_jump_forward_string())
            def rollback(self,n):return self._handle.rollback(n)
            def reset(self):return self._handle.reset()
            def fork(self):
                result=type(self).__new__(type(self));result._handle=self._handle.fork();return result

        _API=SimpleNamespace(Grammar=grammar.Grammar,StructuralTagItem=grammar.StructuralTagItem,
            GrammarCompiler=compiler.GrammarCompiler,TokenizerInfo=TokenizerInfo,
            GrammarMatcher=GrammarMatcher,allocate_token_bitmask=allocate,_native=native)
        return _API

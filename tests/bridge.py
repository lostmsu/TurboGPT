"""ctypes interface to the optional test DLL; never used by training."""
import ctypes as C
import os
from pathlib import Path
import numpy as np

class Config(C.Structure):
    _fields_ = [(x, C.c_int) for x in ('depth', 'context', 'batch', 'device', 'blocks', 'ns_steps')] + [
        (x, C.c_float) for x in ('learning_rate', 'muon_lr', 'beta1', 'beta2', 'momentum',
                                 'weight_decay', 'epsilon', 'clip')] + [('seed', C.c_uint64), ('inflight', C.c_int)]

    def __init__(self, **kwargs):
        super().__init__(4, 4, 16, 0, 0, 5, .0006, .02, .9, .95, .95, .1, 1e-8, 1., 3407, 1)
        for key, value in kwargs.items():
            setattr(self, key, value)
class OneCycle(C.Structure):
    _fields_ = [('total_steps', C.c_int)] + [
        (name, C.c_double) for name in ('warmup_fraction', 'initial_lr_fraction',
                                        'final_lr_fraction', 'low_momentum', 'high_momentum')]

    def __init__(self, total_steps):
        super().__init__(total_steps, .3, .04, .000004, .85, .95)


class StepRates(C.Structure):
    _fields_ = [('multiplier', C.c_float), ('momentum', C.c_float)]


class Native:
    def __init__(self, config):
        self.config = config
        self.lib = C.CDLL(os.environ.get('TURBOGPT_LIBRARY',
                         str(Path(__file__).resolve().parents[1] / 'build/turbogpt.dll')))
        lib = self.lib
        lib.tg_error.restype = C.c_char_p
        lib.tg_create.argtypes = [C.POINTER(Config)]
        lib.tg_create.restype = C.c_void_p
        for name in ('destroy', 'parameter_count', 'blocks', 'shared_bytes', 'synchronize'):
            getattr(lib, 'tg_' + name).argtypes = [C.c_void_p]
        for name in ('set_weights', 'get_weights'):
            getattr(lib, 'tg_' + name).argtypes = [C.c_void_p, C.c_void_p]
        lib.tg_set_data.argtypes = [C.c_void_p, C.c_void_p, C.c_int64]
        lib.tg_train.argtypes = [C.c_void_p, C.c_int]
        lib.tg_training_versions.argtypes = [C.c_void_p, C.c_void_p, C.c_int]
        lib.tg_batch.argtypes = [C.c_void_p] * 6
        lib.tg_optimizer_test.argtypes = [C.c_void_p, C.c_void_p, C.c_int]
        lib.tg_set_onecycle.argtypes = [C.c_void_p, C.POINTER(OneCycle)]
        lib.tg_onecycle_point.argtypes = [C.POINTER(OneCycle), C.c_int, C.POINTER(StepRates)]
        self.handle = lib.tg_create(C.byref(config))
        if not self.handle:
            raise RuntimeError(lib.tg_error().decode())
        self.count = lib.tg_parameter_count(self.handle)

    def check(self, value):
        if value < 0:
            raise RuntimeError(self.lib.tg_error().decode())

    def close(self):
        if self.handle:
            self.lib.tg_destroy(self.handle)
            self.handle = None

    def set_weights(self, w):
        w = np.ascontiguousarray(w, dtype=np.float32)
        assert w.size == self.count
        self.check(self.lib.tg_set_weights(self.handle, w.ctypes.data))

    def weights(self):
        w = np.empty(self.count, np.float32)
        self.check(self.lib.tg_get_weights(self.handle, w.ctypes.data))
        return w

    def batch(self, x, y, backward=True):
        x, y = np.ascontiguousarray(x, dtype=np.uint8), np.ascontiguousarray(y, dtype=np.uint8)
        assert x.shape == y.shape == (self.config.batch, self.config.context)
        logits = np.empty((*x.shape, 256), np.float32)
        grad = np.empty(self.count, np.float32) if backward else None
        losses = np.empty(self.config.context, np.float32)
        self.check(self.lib.tg_batch(self.handle, x.ctypes.data, y.ctypes.data, logits.ctypes.data,
                                     grad.ctypes.data if grad is not None else None, losses.ctypes.data))
        return logits, grad, losses

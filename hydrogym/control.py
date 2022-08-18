from typing import Callable, Optional

from .core import CallbackBase, FlowConfig


# TODO
class ControllerCallback(CallbackBase):
    def __init__(self, callback: Callable, interval: Optional[int] = 1):
        super().__init__(interval=interval)
        self.cb = callback

    def __call__(self, iter: int, t: float, flow: FlowConfig):
        if super().__call__(iter, t, flow):
            self.cb(iter, t, flow)

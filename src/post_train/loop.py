import logging
import weakref
from contextlib import ExitStack
from typing import List, Optional

from detectron2.utils.events import EventStorage
from detectron2.utils.logger import _log_api_usage

from src.tester.hooks import HookTestBase


class Loop:
    """
    Base class for iterative tester with hooks.

    The only assumption we made here is: the test runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.

    Attributes:
        iter(int): the current iteration.

        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.

        max_iter(int): The iteration to end training.

        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self) -> None:
        self._hooks: List[HookTestBase] = []
        self.iter: int = 0
        self.start_iter: int = 0
        self.max_iter: int
        self.storage: EventStorage
        _log_api_usage("tester." + self.__class__.__name__)

    def register_hooks(self, hooks: List[Optional[HookTestBase]]) -> None:
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookTestBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.tester = weakref.proxy(self)
        self._hooks.extend(hooks)

    def loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting loop from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before()
                with ExitStack() as stack:
                    self.apply_stack_context(stack)
                    for self.iter in range(start_iter, max_iter):
                        self.before_step()
                        self.run_step()
                        self.after_step()
                    # self.iter == max_iter can be used by `after_train` to
                    # tell whether the training successfully finished or failed
                    # due to exceptions.
                    self.iter += 1
            except Exception:
                logger.exception("Exception during testing:")
                raise
            finally:
                self.after()

    def apply_stack_context(self, stack : ExitStack):
        pass

    def before(self):
        for h in self._hooks:
            h.before()

    def after(self):
        self.storage.iter = self.iter
        for h in self._hooks:
            h.after()

    def before_step(self):
        # Maintain the invariant that storage.iter == trainer.iter
        # for the entire execution of each step
        self.storage.iter = self.iter

        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()

    def run_step(self):
        raise NotImplementedError

    def state_dict(self):
        ret = {"iteration": self.iter}
        hooks_state = {}
        for h in self._hooks:
            sd = h.state_dict()
            if sd:
                name = type(h).__qualname__
                if name in hooks_state:
                    # TODO handle repetitive stateful hooks
                    continue
                hooks_state[name] = sd
        if hooks_state:
            ret["hooks"] = hooks_state
        return ret

    def load_state_dict(self, state_dict):
        logger = logging.getLogger(__name__)
        self.iter = state_dict["iteration"]
        for key, value in state_dict.get("hooks", {}).items():
            for h in self._hooks:
                try:
                    name = type(h).__qualname__
                except AttributeError:
                    continue
                if name == key:
                    h.load_state_dict(value)
                    break
            else:
                logger.warning(f"Cannot find the hook '{key}', its state_dict is ignored.")
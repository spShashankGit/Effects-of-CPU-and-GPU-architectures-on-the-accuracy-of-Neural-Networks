import sys

from pypads.app.env import InjectionLoggerEnv
from pypads.app.injections.injection import InjectionLogger
from tests.base_test import BaseTest, TEST_FOLDER


def experiment():
    print("I'm an module level experiment")
    return "I'm a return value."


class First(InjectionLogger):
    def __post__(self, ctx, *args, _logger_call, _pypads_pre_return, _pypads_result, _args, _kwargs, **kwargs):
        pass

    def __pre__(ctx, *args, _logger_call: InjectionLoggerEnv, **kwargs):
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()
        print("first")
        pads.cache.run_add(0, True)


class Second(InjectionLogger):
    def __post__(self, ctx, *args, _logger_call, _pypads_pre_return, _pypads_result, _args, _kwargs, **kwargs):
        pass

    def __pre__(ctx, *args, _logger_call: InjectionLoggerEnv, **kwargs):
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()
        print("second")
        if not pads.cache.run_exists(0):
            raise ValueError("Not called as second")
        pads.cache.run_add(1, True)


class Third(InjectionLogger):
    def __post__(self, ctx, *args, _logger_call, _pypads_pre_return, _pypads_result, _args, _kwargs, **kwargs):
        pass

    def __pre__(ctx, *args, _logger_call: InjectionLoggerEnv, **kwargs):
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()
        print("third")
        if not pads.cache.run_exists(1):
            raise ValueError("Not called as third")
        pads.cache.run_add(2, True)


events = {
    "first": First(),
    "second": Second(),
    "third": Third()
}

hooks = {
    "first": {"on": ["order"], "order": 1},
    "second": {"on": ["order"], "order": 2},
    "third": {"on": ["order"], "order": 3},
}

config = {
    "recursion_identity": False,
    "recursion_depth": -1}


class PypadsOrderTest(BaseTest):

    def test_order_lf(self):
        """
        This example will track the experiment exection with the default configuration.
        :return:
        """
        # --------------------------- setup of the tracking ---------------------------
        # Activate tracking of pypads
        from pypads.app.base import PyPads
        tracker = PyPads(uri=TEST_FOLDER, config=config, hooks=hooks, events=events, autostart=True)
        tracker.api.track(experiment, anchors=["order"], ctx=sys.modules[__name__])

        import timeit
        t = timeit.Timer(experiment)
        print(t.timeit(1))

        # --------------------------- asserts ---------------------------
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()
        assert pads.cache.run_exists(0, 1, 2)
        # !-------------------------- asserts ---------------------------

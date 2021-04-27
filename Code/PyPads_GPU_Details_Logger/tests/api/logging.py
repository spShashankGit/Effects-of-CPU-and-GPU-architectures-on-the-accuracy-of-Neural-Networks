import sys

from tests.base_test import BaseTest, TEST_FOLDER
from tests.injections.injection_loggers import RunLogger


def more_experiment(i):
    print("I'm a deep even module level experiment iteration " + str(i))
    return "I'm a deep even return value iteration " + str(i)


def sub_experiment(i):
    print("I'm a deep module level experiment iteration " + str(i))

    if i % 2 == 0:
        more_experiment(i)
    return "I'm a deep return value iteration " + str(i)


def experiment():
    print("I'm an module level experiment")
    for i in range(0, 10):
        sub_experiment(i)
    return "I'm a return value."


logger = RunLogger()

events = {
    "ran_logger": logger
}

hooks = {
    "ran_logger": {"on": ["ran"]},
}

config = {
    "recursion_identity": False,
    "recursion_depth": -1}


class PypadsAPILogging(BaseTest):

    def test_api_track(self):
        """
        This example will track the experiment call and default event and context.
        :return:
        """
        # --------------------------- setup of the tracking ---------------------------
        # Activate tracking of pypads
        from pypads.app.base import PyPads
        tracker = PyPads(uri=TEST_FOLDER, config=config, hooks=hooks, events=events, autostart=True)

        tracker.api.track(experiment, ctx=sys.modules[__name__])

        import timeit
        t = timeit.Timer(experiment)
        print(t.timeit(1))

        # --------------------------- asserts ---------------------------
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()
        assert pads.cache.run_exists(id(logger))
        # !-------------------------- asserts ---------------------------

    def test_api_track_inline(self):
        """
        This example will track the experiment with call and passed event and no context.
        :return:
        """
        # --------------------------- setup of the tracking ---------------------------
        # Activate tracking of pypads
        from pypads.app.base import PyPads
        tracker = PyPads(uri=TEST_FOLDER, config=config, hooks=hooks, events=events, autostart=True)

        def experiment():
            print("I'm an function level experiment")
            return "I'm a return value."

        experiment = tracker.api.track(experiment, anchors=["pypads_log"])

        import timeit
        t = timeit.Timer(experiment)
        print(t.timeit(1))

        # --------------------------- asserts ---------------------------
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()
        assert pads.cache.run_exists(id(logger))
        # !-------------------------- asserts ---------------------------

    def test_api_layered_tracking(self):
        """
        This example will track the experiment execution with the default configuration.
        :return:
        """
        # --------------------------- setup of the tracking ---------------------------
        # Activate tracking of pypads
        from pypads.app.base import PyPads
        tracker = PyPads(uri=TEST_FOLDER, config=config, hooks=hooks, events=events, autostart=True)

        global experiment
        experiment = tracker.api.track(experiment, anchors=["ran"])

        global sub_experiment
        sub_experiment = tracker.api.track(sub_experiment, anchors=["ran"])

        global more_experiment
        more_experiment = tracker.api.track(more_experiment, anchors=["ran"])

        import timeit
        t = timeit.Timer(experiment)
        print(t.timeit(1))

        # --------------------------- asserts ---------------------------
        from pypads.app.pypads import get_current_pads
        pads = get_current_pads()
        assert pads.cache.run_exists(id(logger))
        assert pads.cache.run_get(id(logger)) == 16
        # !-------------------------- asserts ---------------------------
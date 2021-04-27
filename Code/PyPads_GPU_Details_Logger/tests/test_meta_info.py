from pypads.model.logger_output import MetricMetaModel, ParameterMetaModel, ArtifactMetaModel
from pypads.model.models import get_reference
from pypads.utils.logging_util import FileFormats
from tests.base_test import BaseTest, TEST_FOLDER


class PypadsHookTest(BaseTest):

    def test_track_param(self):

        # --------------------------- setup of the tracking ---------------------------
        # Activate tracking of pypads
        from pypads.app.base import PyPads

        name = 'networks_shape'
        neural_network_shape = [10, 10, 3]
        description = 'Shape of the fully connected network'
        keys = ['experiment_id', 'run_id', 'category', 'storage_type']

        tracker = PyPads(uri=TEST_FOLDER, autostart=True)
        tracker.api.log_param(key=name, value=str(neural_network_shape), description=description)

        holder = tracker.api.get_programmatic_output()
        meta = ParameterMetaModel(name=name, value_format='str', data=str(neural_network_shape),
                                  description=description, parent=holder,
                                  produced_by=holder.produced_by,
                                  part_of=get_reference(holder))

        # --------------------------- asserts ---------------------------
        # Number of retrieved items should be 1
        retrieved_items = [x for x in tracker.results.get_parameters(name='networks_shape', run_id=meta.run.uid)]
        assert len(retrieved_items) == 1

        retrieved_items = retrieved_items[0]
        for key in keys:
            assert retrieved_items.dict().get(key) == meta.dict().get(key)
        # !-------------------------- asserts ---------------------------

    def test_track_metric(self):
        # --------------------------- setup of the tracking ---------------------------
        # Activate tracking of pypads
        from pypads.app.base import PyPads

        name = "some_metric"
        description = 'Some description'
        value = 1
        step = 0
        keys = ['experiment_id', 'run_id', 'category', 'storage_type', 'description', 'name', 'data']

        tracker = PyPads(uri=TEST_FOLDER)
        tracker.activate_tracking()
        tracker.start_track(experiment_name='TEST CASE EXPERIMENT')
        # meta = MetricMetaModel(url='https://some.metric.url', name='some_metric', description='some description',
        #                        step=0)
        tracker.api.log_metric(name, value=value, description=description, step=step)

        holder = tracker.api.get_programmatic_output()
        meta = MetricMetaModel(name=name, value_format='str', data=str(value), step=step,
                               description=description, parent=holder,
                               produced_by=holder.produced_by,
                               part_of=get_reference(holder))

        artifacts = [x for x in tracker.results.get_metrics(experiment_name='TEST CASE EXPERIMENT',
                                                            name=name, step=step, run_id=meta.run.uid)]
        # --------------------------- asserts ---------------------------
        assert len(artifacts) == 1
        for key in keys:
            assert artifacts[0].dict().get(key) == meta.dict().get(key)

        # !-------------------------- asserts ---------------------------

    def test_track_mem_artifact(self):
        # --------------------------- setup of the tracking ---------------------------
        # Activate tracking of pypads
        from pypads.app.base import PyPads
        tracker = PyPads(uri=TEST_FOLDER, autostart=True)

        path = 'some_artifact'
        description = 'Storing test array as an artifact'

        keys = ['experiment_id', 'run_id', 'category', 'storage_type', 'description', 'name', 'description',
                'produced_by']

        import numpy as np
        obj = np.random.random(size=(3, 3))

        holder = tracker.api.get_programmatic_output()

        tracker.api.log_mem_artifact(path=path, obj=obj, write_format=FileFormats.pickle, description=description,
                                     additional_data=None, holder=None)

        meta = ArtifactMetaModel(value_format='str', file_format=FileFormats.pickle,
                                 description=description, file_size=229,
                                 data=str(obj),
                                 parent=holder,
                                 produced_by=holder.produced_by,
                                 part_of=get_reference(holder))

        artifacts = [x for x in tracker.results.get_artifacts(run_id=meta.run.uid) if x.data == path + '.pickle']

        # --------------------------- asserts ---------------------------
        assert len(artifacts) == 1
        artifacts = artifacts[0]
        for key in keys:
            assert meta.dict().get(key) == artifacts.dict().get(key)
        # !-------------------------- asserts ---------------------------

    def test_track_artifact(self):
        # --------------------------- setup of the tracking ---------------------------
        # Activate tracking of pypads
        from pypads.app.base import PyPads
        tracker = PyPads(uri=TEST_FOLDER, autostart=True)

        import os
        import pickle as pkl
        import numpy as np

        name = 'some_artifact.pickle'
        path = os.path.join(os.getcwd(), name)
        description = 'logging an artifact from a local path'
        obj = np.random.random(size=(3, 3))

        keys = ['experiment_id', 'run_id', 'category', 'storage_type', 'description', 'name', 'description',
                'produced_by']

        with open(path, 'wb') as f:
            pkl.dump(obj, f)

        tracker.api.log_artifact(local_path=path, additional_data=None, holder=None, description=description)
        os.remove(path=path)

        holder = tracker.api.get_programmatic_output()
        meta = ArtifactMetaModel(value_format='str', file_format=FileFormats.pickle,
                                 description=description, file_size=229,
                                 data=str(obj),
                                 parent=holder,
                                 produced_by=holder.produced_by,
                                 part_of=get_reference(holder))

        # Load the artifacts
        artifacts = [x for x in tracker.results.get_artifacts(run_id=meta.run.uid) if x.data == name]

        # Load the data from the pypads path
        loaded_data = tracker.results.load_artifact(name, read_format=FileFormats.pickle)

        # --------------------------- asserts ---------------------------
        assert bool((loaded_data == obj).all()) is True
        assert len(artifacts) == 1

        artifacts = artifacts[0]
        for key in keys:
            assert meta.dict().get(key) == artifacts.dict().get(key)
        # !-------------------------- asserts ---------------------------

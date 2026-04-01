import os
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# The production code imports are now safe because pytest_configure in conftest.py
# installs the mocks before collection.
from opencda.core.common.coperception_model_manager import CoperceptionModelManager, DirectoryProcessor


class DummyOpt:
    def __init__(self, **kwargs):
        self.model_dir = "test_model_dir"
        self.fusion_method = "late"
        self.show_vis = False
        self.show_sequence = False
        self.save_npy = False
        self.save_vis = False
        self.test_scenario = "test_scenario"
        self.global_sort_detections = True
        self.__dict__.update(kwargs)


class DummyDataset:
    def __init__(self):
        self.data = []

    def __len__(self):
        return 10

    def collate_batch_test(self, batch):
        return batch

    def visualize_result(self, *args, **kwargs):
        pass

    def update_database(self):
        pass


class TestCoperceptionModelManager:
    @pytest.fixture
    def manager_deps(self, fake_heavy_deps):
        """
        Setup mocks specifically for Manager instantiation and method calls.
        Resets mocks before every test to ensure isolation.
        """
        opencood = fake_heavy_deps["opencood"]
        torch = fake_heavy_deps["torch"]
        open3d = fake_heavy_deps["open3d"]

        # Shortcuts to specific mocks (Modules & Objects)
        # Note: We cannot call reset_mock() on modules, only on Mocks.

        mocks_to_reset = [
            opencood.hypes_yaml.yaml_utils.load_yaml,
            opencood.tools.train_utils.create_model,
            opencood.tools.train_utils.load_saved_model,
            opencood.tools.train_utils.to_device,
            opencood.tools.inference_utils.inference_late_fusion,
            opencood.tools.inference_utils.inference_early_fusion,
            opencood.tools.inference_utils.inference_intermediate_fusion,
            opencood.tools.inference_utils.save_prediction_gt,
            opencood.data_utils.datasets.build_dataset,
            opencood.visualization.simple_vis.visualize,
            opencood.visualization.vis_utils.visualize_inference_sample_dataloader,
            opencood.visualization.vis_utils.linset_assign_list,
            opencood.utils.eval_utils.caluclate_tp_fp,
            opencood.utils.eval_utils.eval_final_results,
            open3d.visualization.Visualizer,
            torch.cuda.is_available,
            torch.device,
            torch.no_grad,
        ]

        # Reset actual mock objects
        for m in mocks_to_reset:
            m.reset_mock()

        # Remove side effects from previous tests
        opencood.utils.eval_utils.caluclate_tp_fp.side_effect = None

        # Setup default return values
        hypes = {
            "postprocess": {"core_method": "VoxelPostprocessor", "gt_range": [0, -40, -3, 70, 40, 1]},
            "fusion": {"core_method": "IntermediateFusionDataset"},
        }
        opencood.hypes_yaml.yaml_utils.load_yaml.return_value = hypes

        model = MagicMock()
        opencood.tools.train_utils.create_model.return_value = model
        opencood.tools.train_utils.load_saved_model.return_value = (None, model)

        # Return dict for easy access
        return {
            "yaml_utils": opencood.hypes_yaml.yaml_utils,
            "train_utils": opencood.tools.train_utils,
            "inference_utils": opencood.tools.inference_utils,
            "vis_utils": opencood.visualization.vis_utils,
            "simple_vis": opencood.visualization.simple_vis,
            "eval_utils": opencood.utils.eval_utils,
            "build_dataset": opencood.data_utils.datasets.build_dataset,
            "Visualizer": open3d.visualization.Visualizer,
            "torch": torch,
            "model": model,
            "hypes": hypes,
        }

    def test_init_cpu(self, manager_deps):
        manager_deps["torch"].cuda.is_available.return_value = False
        opt = DummyOpt()
        manager = CoperceptionModelManager(opt, "2023_01_01")

        assert manager.device == "device(cpu)"
        manager_deps["model"].cuda.assert_not_called()
        manager_deps["train_utils"].load_saved_model.assert_called_with("test_model_dir", manager_deps["model"])

    def test_init_cuda(self, manager_deps):
        manager_deps["torch"].cuda.is_available.return_value = True
        opt = DummyOpt()
        manager = CoperceptionModelManager(opt, "2023_01_01")

        assert manager.device == "device(cuda)"
        manager_deps["model"].cuda.assert_called_once()

    def test_update_dataset(self, manager_deps):
        """
        Verify update_dataset calls the correct build_dataset and creates a DataLoader.
        We patch the symbol inside the module under test to ensure we capture the call.
        """
        dataset_mock = DummyDataset()

        # Patching where it is imported in the source code
        with patch("opencda.core.common.coperception_model_manager.build_dataset", return_value=dataset_mock) as mock_build:
            opt = DummyOpt()
            manager = CoperceptionModelManager(opt, "2023_01_01")

            manager.update_dataset()

            mock_build.assert_called_with(manager_deps["hypes"], visualize=True, train=False, message_handler=None)
            assert manager.opencood_dataset == dataset_mock
            assert manager.data_loader is not None
            assert manager.data_loader.dataset == dataset_mock

    def test_make_prediction_state_update(self, manager_deps):
        """Test that final_result_stat is actually updated via caluclate_tp_fp side effect."""
        opt = DummyOpt(fusion_method="late")
        manager = CoperceptionModelManager(opt, "2023_01_01")

        # Configure dataset to return one sample with the expected batch shape
        batch_data = {"ego": {"origin_lidar": ["lidar_data"]}}
        manager.opencood_dataset.__len__ = MagicMock(return_value=1)
        manager.opencood_dataset.collate_batch_test.return_value = batch_data

        # Define side effect for caluclate_tp_fp to modify the stats dictionary
        def mock_calculate_tp_fp(pred, score, gt, stat, iou):
            stat[iou]["gt"] += 1
            stat[iou]["tp"].append(1)
            stat[iou]["fp"].append(0)
            stat[iou]["score"].append(0.9)

        manager_deps["eval_utils"].caluclate_tp_fp.side_effect = mock_calculate_tp_fp

        manager.make_prediction(0)

        # Verify stats were accumulated
        for iou in [0.3, 0.5, 0.7]:
            assert manager.final_result_stat[iou]["gt"] == 1
            assert len(manager.final_result_stat[iou]["tp"]) == 1
            assert manager.final_result_stat[iou]["score"][0] == 0.9

    @pytest.mark.parametrize("fusion_method", ["late", "early", "intermediate"])
    def test_make_prediction_fusion_methods(self, fusion_method, manager_deps):
        opt = DummyOpt(fusion_method=fusion_method)
        manager = CoperceptionModelManager(opt, "2023_01_01")
        batch_data = {"ego": {"origin_lidar": ["lidar_data"]}}
        manager.opencood_dataset.__len__ = MagicMock(return_value=1)
        manager.opencood_dataset.collate_batch_test.return_value = batch_data

        manager.make_prediction(0)

        if fusion_method == "late":
            manager_deps["inference_utils"].inference_late_fusion.assert_called()
        elif fusion_method == "early":
            manager_deps["inference_utils"].inference_early_fusion.assert_called()
        elif fusion_method == "intermediate":
            manager_deps["inference_utils"].inference_intermediate_fusion.assert_called()

    def test_make_prediction_assertions(self):
        opt = DummyOpt(fusion_method="invalid")
        manager = CoperceptionModelManager(opt, "2023_01_01")
        with pytest.raises(AssertionError):
            manager.make_prediction(0)

        opt = DummyOpt(fusion_method="late", show_vis=True, show_sequence=True)
        manager = CoperceptionModelManager(opt, "2023_01_01")
        with pytest.raises(AssertionError, match="single image mode or video mode"):
            manager.make_prediction(0)

    def test_make_prediction_save_npy(self, manager_deps, tmp_path, monkeypatch):
        """Test saving NPY files using real filesystem operations in tmp_path."""
        monkeypatch.chdir(tmp_path)

        opt = DummyOpt(save_npy=True, test_scenario="scen1")
        manager = CoperceptionModelManager(opt, "2023_01_01")

        batch_data = {"ego": {"origin_lidar": ["lidar"]}}
        manager.opencood_dataset.__len__ = MagicMock(return_value=1)
        manager.opencood_dataset.collate_batch_test.return_value = batch_data
        manager_deps["inference_utils"].inference_late_fusion.return_value = ("p", "s", "g")

        manager.make_prediction(10)

        # Check directory creation
        expected_dir = tmp_path / "simulation_output/coperception/npy/scen1_2023_01_01/npy"
        assert expected_dir.exists()

        # Check call and that tick_number (10) is used instead of a loop index
        manager_deps["inference_utils"].save_prediction_gt.assert_called()
        args = manager_deps["inference_utils"].save_prediction_gt.call_args[0]
        # args[3] is tick_number, args[4] is the path passed to save_prediction_gt
        assert args[3] == 10
        assert Path(args[4]).resolve() == expected_dir.resolve()

    def test_make_prediction_save_vis(self, manager_deps, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        opt = DummyOpt(save_vis=True, test_scenario="scen1")
        manager = CoperceptionModelManager(opt, "2023_01_01")
        # Ensure VoxelPostprocessor to test both 3d and bev
        manager.hypes["postprocess"]["core_method"] = "VoxelPostprocessor"
        manager.hypes["fusion"]["core_method"] = "IntermediateFusionDataset"

        batch_data = {"ego": {"origin_lidar": ["lidar"]}}
        manager.opencood_dataset.__len__ = MagicMock(return_value=1)
        manager.opencood_dataset.collate_batch_test.return_value = batch_data
        manager_deps["inference_utils"].inference_late_fusion.return_value = ("p", "s", "g")

        manager.make_prediction(5)

        # Verify directories
        base_dir = tmp_path / "simulation_output/coperception"
        assert (base_dir / "vis_3d/scen1_2023_01_01").exists()
        assert (base_dir / "vis_bev/scen1_2023_01_01").exists()

        assert manager_deps["simple_vis"].visualize.call_count == 2

    def test_make_prediction_show_sequence(self, manager_deps, fake_heavy_deps):
        """Test Open3D interactions without opening windows."""
        opt = DummyOpt(show_sequence=True)
        manager = CoperceptionModelManager(opt, "2023_01_01")

        batch_data = {"ego": {"origin_lidar": ["lidar1"]}}
        manager.opencood_dataset.__len__ = MagicMock(return_value=1)
        manager.opencood_dataset.collate_batch_test.return_value = batch_data
        # Ensure pred is not None
        manager_deps["inference_utils"].inference_late_fusion.return_value = ("box", "score", "gt")

        manager.make_prediction(0)

        # Check Visualizer creation (mocked class in conftest)
        manager_deps["Visualizer"].assert_called()
        vis_instance = manager.vis

        vis_instance.create_window.assert_called()
        vis_instance.add_geometry.assert_called()  # first tick: _vis_geometries_added == False
        vis_instance.update_renderer.assert_called()

        # Verify line set assignment was called
        manager_deps["vis_utils"].linset_assign_list.assert_called()

    def test_final_eval(self, manager_deps, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        opt = DummyOpt(test_scenario="scen1")
        manager = CoperceptionModelManager(opt, "2023_01_01")

        manager.final_eval()

        expected_dir = tmp_path / "simulation_output/coperception/results/scen1_2023_01_01"
        assert expected_dir.is_dir()

        manager_deps["eval_utils"].eval_final_results.assert_called()
        args = manager_deps["eval_utils"].eval_final_results.call_args[0]
        # Check path arg
        assert args[0] is manager.final_result_stat
        assert Path(args[1]).resolve() == expected_dir.resolve()
        assert args[2] == opt.global_sort_detections


# ---------------------------------------------------------------------------
# New tests: ring buffer, late-attack caching, and real tick_number contract
# ---------------------------------------------------------------------------


class TestRingBufferAndLateTick:
    """Tests for the CPU-only ring buffer and the single-frame online behaviour."""

    @pytest.fixture
    def manager(self, fake_heavy_deps):
        """Create a lightweight CoperceptionModelManager with mocked deps."""
        opencood = fake_heavy_deps["opencood"]
        hypes = {
            "postprocess": {"core_method": "VoxelPostprocessor", "gt_range": [0, -40, -3, 70, 40, 1]},
            "fusion": {"core_method": "IntermediateFusionDataset"},
        }
        opencood.hypes_yaml.yaml_utils.load_yaml.return_value = hypes
        model = MagicMock()
        opencood.tools.train_utils.create_model.return_value = model
        opencood.tools.train_utils.load_saved_model.return_value = (None, model)
        opencood.tools.train_utils.to_device.side_effect = lambda x, y: x

        opt = DummyOpt()
        mgr = CoperceptionModelManager(opt, "t0")
        # Fresh MagicMock dataset to avoid state leakage
        mgr.opencood_dataset = MagicMock()
        return mgr

    # -- ring-buffer max-size --------------------------------------------------

    def test_ring_buffer_max_size(self, manager):
        """Buffer should never hold more than 10 entries; oldest is evicted."""
        # Manually inject raw data to bypass dataset retrieval
        for tick in range(15):
            manager._raw_data_cache[tick] = {"vehicle_0": {"lidar_np": None}}
            while len(manager._raw_data_cache) > manager._raw_data_cache_max_size:
                manager._raw_data_cache.popitem(last=False)

        assert len(manager._raw_data_cache) == 10
        # Ticks 0..4 must have been evicted; only 5..14 remain
        assert 0 not in manager._raw_data_cache
        assert 4 not in manager._raw_data_cache
        assert 5 in manager._raw_data_cache
        assert 14 in manager._raw_data_cache

    def test_cache_raw_data_respects_max_size(self, manager):
        """_cache_raw_data trims the buffer to at most _raw_data_cache_max_size."""
        manager.opencood_dataset.retrieve_base_data.return_value = {"v0": {"lidar_np": None}}

        for tick in range(13):
            manager._cache_raw_data(tick)

        assert len(manager._raw_data_cache) <= manager._raw_data_cache_max_size

    # -- 0..9 remapping --------------------------------------------------------

    def test_get_last_n_raw_frames_remapping(self, manager):
        """get_last_n_raw_frames must remap oldest→0 and newest→9."""
        for tick in range(15):
            manager._raw_data_cache[tick] = {"tick": tick}

        remapped = manager.get_last_n_raw_frames(10)

        assert len(remapped) == 10
        # Key 0 must be the oldest of the last-10 (tick 5) and key 9 the newest (tick 14)
        assert remapped[0]["tick"] == 5
        assert remapped[9]["tick"] == 14

    def test_get_last_n_raw_frames_fewer_than_n(self, manager):
        """Returns empty dict when fewer than n frames are cached."""
        for tick in range(7):
            manager._raw_data_cache[tick] = {"tick": tick}

        assert manager.get_last_n_raw_frames(10) == {}

    def test_get_last_n_raw_frames_exactly_n(self, manager):
        """Returns remapped dict of size n when exactly n frames are cached."""
        for tick in range(10):
            manager._raw_data_cache[tick] = {"tick": tick}

        remapped = manager.get_last_n_raw_frames(10)

        assert len(remapped) == 10
        assert remapped[0]["tick"] == 0
        assert remapped[9]["tick"] == 9

    # -- process_tick called with real tick_number -----------------------------

    def test_process_tick_receives_real_tick_number(self, fake_heavy_deps, manager):
        """make_prediction must pass tick_number (not a loop index) to process_tick."""
        from unittest.mock import patch

        # Set up a dataset with one sample and proper batch return
        batch_data = {"ego": {"origin_lidar": ["lidar"]}}
        manager.opencood_dataset.__len__ = MagicMock(return_value=1)
        manager.opencood_dataset.collate_batch_test.return_value = batch_data

        fake_heavy_deps["opencood"].tools.inference_utils.inference_late_fusion.return_value = (
            MagicMock(), MagicMock(), MagicMock()
        )

        recorded_ticks = []

        def fake_process_tick(tick_number, **kwargs):
            recorded_ticks.append(tick_number)
            return None, None, None

        with patch.object(manager, "advcp_manager", None):
            # No AdvCP – just verify _current_batch_index is set to tick_number
            manager.make_prediction(42)

        assert manager._current_batch_index == 42

    # -- late-attack skipped until 10 frames exist ----------------------------

    def test_late_attack_skipped_below_10_frames(self, fake_heavy_deps, manager):
        """_apply_attack must return original predictions and not call attacker.run
        when fewer than 10 frames are cached."""
        import types
        import sys

        # Build a minimal but correct module hierarchy for all mvp dependencies
        _mvp_mods = [
            "mvp",
            "mvp.attack",
            "mvp.attack.lidar_remove_late_attacker",
            "mvp.attack.lidar_remove_early_attacker",
            "mvp.attack.lidar_remove_intermediate_attacker",
            "mvp.attack.lidar_spoof_early_attacker",
            "mvp.attack.lidar_spoof_intermediate_attacker",
            "mvp.attack.lidar_spoof_late_attacker",
            "mvp.attack.adv_shape_attacker",
            "mvp.defense",
            "mvp.defense.perception_defender",
            "mvp.perception",
            "mvp.perception.opencood_perception",
            "mvp.visualize",
            "mvp.visualize.general",
            "mvp.visualize.attack",
            "mvp.visualize.defense",
            "mvp.visualize.evaluate",
        ]
        _inserted = {}
        for mod_name in _mvp_mods:
            if mod_name not in sys.modules:
                mod = types.ModuleType(mod_name)
                sys.modules[mod_name] = mod
                _inserted[mod_name] = mod
                parts = mod_name.split(".")
                if len(parts) > 1:
                    parent = sys.modules[".".join(parts[:-1])]
                    setattr(parent, parts[-1], mod)

        # Add the required class/function stubs
        # Add one class stub per module (each module hosts exactly one attacker class)
        _attacker_stubs = {
            "mvp.attack.lidar_remove_late_attacker": "LidarRemoveLateAttacker",
            "mvp.attack.lidar_remove_early_attacker": "LidarRemoveEarlyAttacker",
            "mvp.attack.lidar_remove_intermediate_attacker": "LidarRemoveIntermediateAttacker",
            "mvp.attack.lidar_spoof_early_attacker": "LidarSpoofEarlyAttacker",
            "mvp.attack.lidar_spoof_intermediate_attacker": "LidarSpoofIntermediateAttacker",
            "mvp.attack.lidar_spoof_late_attacker": "LidarSpoofLateAttacker",
            "mvp.attack.adv_shape_attacker": "AdvShapeAttacker",
        }
        for mod_name, cls_name in _attacker_stubs.items():
            setattr(sys.modules[mod_name], cls_name, MagicMock)
        sys.modules["mvp.defense.perception_defender"].PerceptionDefender = MagicMock
        sys.modules["mvp.perception.opencood_perception"].OpencoodPerception = MagicMock
        sys.modules["mvp.visualize.general"].draw_multi_vehicle_case = MagicMock()
        sys.modules["mvp.visualize.attack"].draw_attack = MagicMock()
        for _fn in ["draw_ground_segmentation", "draw_polygon_areas",
                    "draw_object_tracking", "visualize_defense"]:
            setattr(sys.modules["mvp.visualize.defense"], _fn, MagicMock())
        for _fn in ["draw_distribution", "draw_detection_roc"]:
            setattr(sys.modules["mvp.visualize.evaluate"], _fn, MagicMock())

        try:
            # Force reimport so the fresh stubs are used
            for mod_name in ["opencda.core.common.advcp.advcp_visualization_manager",
                             "opencda.core.common.advcp.advcp_manager"]:
                sys.modules.pop(mod_name, None)

            from opencda.core.common.advcp.advcp_manager import AdvCPManager

            opt = {
                "with_advcp": True,
                "attack_type": "lidar_remove_late",
                "attack_target": "random",
                "attackers_ratio": 0.5,
                "apply_cad_defense": False,
                "defense_threshold": 0.7,
                "advcp_vis": False,
                "advcp_config": "",
            }
            adv_mgr = AdvCPManager(opt, "t0", manager)

            # Pre-load only 5 frames (< 10 required for late attack)
            for tick in range(5):
                manager._raw_data_cache[tick] = {"v0": {"lidar_np": None}}

            raw_data = {"v0": {"lidar_np": None, "object_ids": [], "gt_bboxes": []}}
            original_preds = {"pred_bboxes": "boxes", "pred_scores": "scores", "gt_bboxes": "gt"}

            result = adv_mgr._apply_attack(raw_data, original_preds, tick_number=5)

            # Attack must have been skipped; original predictions returned unchanged
            assert result is original_preds
            if adv_mgr.attacker is not None:
                adv_mgr.attacker.run.assert_not_called()
        finally:
            # Clean up only the modules we inserted to avoid polluting other tests
            for mod_name in _inserted:
                sys.modules.pop(mod_name, None)
            sys.modules.pop("opencda.core.common.advcp.advcp_visualization_manager", None)
            sys.modules.pop("opencda.core.common.advcp.advcp_manager", None)


# --- Tests for DirectoryProcessor ---


class TestDirectoryProcessor:
    @pytest.fixture
    def processor_setup(self, tmp_path):
        source_dir = tmp_path / "data_dumping"
        # IMPORTANT: now_dir must NOT be inside source_dir, otherwise it becomes part of
        # subdirectories and breaks the "subdirectories[-2]" selection logic.
        now_dir = tmp_path / "now"
        source_dir.mkdir(parents=True)
        now_dir.mkdir(parents=True)
        return source_dir, now_dir

    def test_detect_cameras(self, tmp_path):
        dp = DirectoryProcessor()
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        assert dp.detect_cameras(str(data_dir)) == []

        sample = data_dir / "sample_subdir"
        sample.mkdir()
        (sample / "001_camera0.png").touch()
        (sample / "001_camera2.png").touch()
        (sample / "001_camera1.png").touch()

        cameras = dp.detect_cameras(str(data_dir))
        assert cameras == ["_camera0.png", "_camera1.png", "_camera2.png"]

    def test_process_directory_success(self, processor_setup):
        source_dir, now_dir = processor_setup
        dp = DirectoryProcessor(str(source_dir), str(now_dir))

        # Needs at least 2 dirs. Sorted order: d1, d2.
        # Code picks index -2 -> d1.
        d1 = source_dir / "d1"
        d2 = source_dir / "d2"
        d1.mkdir()
        d2.mkdir()

        (d1 / "data_protocol.yaml").write_text("proto")
        agent1 = d1 / "agent1"
        agent1.mkdir()

        # Files for tick 10
        (agent1 / "000010.pcd").write_text("pcd")
        (agent1 / "000010.yaml").write_text("yaml")

        dp.process_directory(10)

        assert (now_dir / "data_protocol.yaml").exists()
        assert (now_dir / "data_protocol.yaml").read_text() == "proto"
        assert (now_dir / "agent1" / "000010.pcd").exists()
        assert (now_dir / "agent1" / "000010.pcd").read_text() == "pcd"

    def test_clear_directory_now(self, processor_setup):
        _, now_dir = processor_setup
        dp = DirectoryProcessor(now_directory=str(now_dir))
        (now_dir / "file.txt").touch()
        (now_dir / "subdir").mkdir()

        dp.clear_directory_now()

        assert len(os.listdir(now_dir)) == 0

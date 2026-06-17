from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_DIR = PROJECT_ROOT / "UI"
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

from retraining_runtime import gpu_diagnostics as diagnostics
from retraining_runtime import repair_preflight
from retraining_runtime.ui_policy import (
    BLOCKED,
    CPU_FALLBACK,
    REPAIR_OR_CPU,
    retraining_start_policy,
)


class _FakeTensor:
    def __mul__(self, value):
        return self


class _FakeCuda:
    def __init__(self, available: bool, allocation_fails: bool = False):
        self.available = available
        self.allocation_fails = allocation_fails

    def is_available(self):
        return self.available

    def device_count(self):
        return 1 if self.available else 0

    def get_device_name(self, index):
        return "Test NVIDIA GPU"

    def synchronize(self):
        return None


def _fake_torch(cuda_available=True, cuda_version="11.8", allocation_fails=False):
    module = types.ModuleType("torch")
    module.__version__ = "2.0.1+cu118" if cuda_version else "2.0.1+cpu"
    module.version = types.SimpleNamespace(cuda=cuda_version)
    module.cuda = _FakeCuda(cuda_available, allocation_fails)
    module.float32 = object()

    def tensor(values, device="cpu", **kwargs):
        if device == "cuda" and allocation_fails:
            raise RuntimeError("test allocation failure")
        return _FakeTensor()

    module.tensor = tensor
    return module


class DiagnosticStateTests(unittest.TestCase):
    def diagnose(
        self,
        *,
        hardware=True,
        cuda_available=True,
        torch_cuda="11.8",
        allocation_fails=False,
        detectron_native=True,
        detectron_cuda=True,
        detectron_cuda_version="11.8",
        detectron_cpu_error=None,
        detectron_gpu_error=None,
        torchvision_cpu_error=None,
        torchvision_gpu_error=None,
    ):
        torch_module = _fake_torch(cuda_available, torch_cuda, allocation_fails)

        def torchvision_probe(_torch, device="cpu"):
            error = torchvision_gpu_error if device == "cuda" else torchvision_cpu_error
            return ("0.15.2+cu118" if error is None else None, error)

        def detectron_native_probe(_torch, device="cpu"):
            return detectron_gpu_error if device == "cuda" else detectron_cpu_error

        detectron_error = None if detectron_native else "missing detectron2._C"
        with patch.dict(sys.modules, {"torch": torch_module}), patch.object(
            diagnostics,
            "_detect_nvidia_gpus",
            return_value=(hardware, ["Test NVIDIA GPU"] if hardware else [], None),
        ), patch.object(
            diagnostics,
            "_probe_torchvision",
            side_effect=torchvision_probe,
        ), patch.object(
            diagnostics,
            "_probe_detectron2",
            return_value=(
                "0.6",
                detectron_native,
                detectron_cuda if detectron_native else None,
                detectron_cuda_version if detectron_native else None,
                detectron_error,
            ),
        ), patch.object(
            diagnostics,
            "_probe_detectron2_native_op",
            side_effect=detectron_native_probe,
        ):
            return diagnostics.diagnose_gpu_support()

    def test_hardware_missing_uses_cpu(self):
        report = self.diagnose(hardware=False, cuda_available=False, detectron_cuda=False)
        self.assertEqual(report.status, diagnostics.HARDWARE_MISSING)
        self.assertTrue(report.cpu_ready)
        self.assertFalse(report.gpu_ready)
        self.assertEqual(report.selected_device, "cpu")

    def test_cpu_only_stack_uses_cpu(self):
        report = self.diagnose(cuda_available=False, torch_cuda=None, detectron_cuda=False)
        self.assertEqual(report.status, diagnostics.HARDWARE_PRESENT_SOFTWARE_MISSING)
        self.assertTrue(report.retraining_available)
        self.assertEqual(report.selected_device, "cpu")

    def test_cuda_ready_selects_gpu(self):
        report = self.diagnose()
        self.assertEqual(report.status, diagnostics.GPU_READY)
        self.assertTrue(report.gpu_ready)
        self.assertEqual(report.selected_device, "cuda")

    def test_cuda_allocation_failure_falls_back_to_cpu(self):
        report = self.diagnose(allocation_fails=True)
        self.assertEqual(report.status, diagnostics.CUDA_BROKEN)
        self.assertTrue(report.cpu_ready)
        self.assertEqual(report.selected_device, "cpu")

    def test_detectron2_cpu_only_falls_back_to_cpu(self):
        report = self.diagnose(detectron_cuda=False, detectron_cuda_version=None)
        self.assertEqual(report.status, diagnostics.HARDWARE_PRESENT_SOFTWARE_MISSING)
        self.assertTrue(report.cpu_ready)
        self.assertFalse(report.gpu_ready)

    def test_detectron2_cpu_native_failure_blocks_retraining(self):
        report = self.diagnose(detectron_cpu_error="CPU native op unavailable")
        self.assertEqual(report.status, diagnostics.RETRAINING_RUNTIME_BROKEN)
        self.assertFalse(report.cpu_ready)

    def test_detectron2_gpu_architecture_failure_falls_back_to_cpu(self):
        error = "Detectron2 native operation failed on cuda: no kernel image is available"
        report = self.diagnose(detectron_gpu_error=error)
        self.assertEqual(report.status, diagnostics.CUDA_BROKEN)
        self.assertTrue(report.cpu_ready)
        self.assertFalse(report.gpu_ready)
        self.assertEqual(report.reason, error)

    def test_missing_detectron2_native_extension_blocks_retraining(self):
        report = self.diagnose(detectron_native=False)
        self.assertEqual(report.status, diagnostics.RETRAINING_RUNTIME_BROKEN)
        self.assertFalse(report.retraining_available)
        with self.assertRaises(RuntimeError):
            diagnostics.select_training_device(report)

    def test_cuda_version_mismatch_falls_back_to_cpu(self):
        report = self.diagnose(detectron_cuda_version="12.1")
        self.assertEqual(report.status, diagnostics.CUDA_BROKEN)
        self.assertTrue(report.cpu_ready)
        self.assertFalse(report.gpu_ready)

    def test_torchvision_gpu_failure_falls_back_to_cpu(self):
        report = self.diagnose(torchvision_gpu_error="CUDA NMS unavailable")
        self.assertEqual(report.status, diagnostics.CUDA_BROKEN)
        self.assertTrue(report.cpu_ready)
        self.assertFalse(report.gpu_ready)
        self.assertEqual(report.reason, "CUDA NMS unavailable")

    def test_torchvision_cpu_failure_blocks_retraining(self):
        report = self.diagnose(torchvision_cpu_error="CPU NMS unavailable")
        self.assertEqual(report.status, diagnostics.RETRAINING_RUNTIME_BROKEN)
        self.assertFalse(report.cpu_ready)


class PackagedPolicyTests(unittest.TestCase):
    def report(self, **overrides):
        values = {
            "status": diagnostics.CUDA_BROKEN,
            "selected_device": "cpu",
            "hardware_present": True,
            "retraining_available": True,
            "cpu_ready": True,
            "gpu_ready": False,
            "repair_recommended": True,
        }
        values.update(overrides)
        return diagnostics.GpuDiagnostic(**values)

    def test_frozen_gpu_failure_only_offers_cpu_fallback_policy(self):
        self.assertEqual(retraining_start_policy(self.report(), frozen=True), CPU_FALLBACK)

    def test_source_gpu_failure_offers_repair_policy(self):
        self.assertEqual(retraining_start_policy(self.report(), frozen=False), REPAIR_OR_CPU)

    def test_broken_frozen_runtime_is_blocked(self):
        report = self.report(
            status=diagnostics.RETRAINING_RUNTIME_BROKEN,
            retraining_available=False,
            cpu_ready=False,
        )
        self.assertEqual(retraining_start_policy(report, frozen=True), BLOCKED)


class PackagingReadinessContractTests(unittest.TestCase):
    def test_diagnostic_entrypoint_runs_before_heavy_ui_imports(self):
        source = (UI_DIR / "PolyVisionMain.py").read_text(encoding="utf-8")
        diagnostic_guard = source.index('"--diagnose-retraining" in sys.argv[1:]')
        heavy_import = source.index("from PIL import Image")
        self.assertLess(diagnostic_guard, heavy_import)

    def test_retraining_ui_does_not_dynamically_load_external_python(self):
        source = (UI_DIR / "Retrain.py").read_text(encoding="utf-8")
        self.assertNotIn("spec_from_file_location", source)
        self.assertIn("from retraining_runtime import train as retraining_train", source)

    def test_repair_preflight_runs_before_active_pip_changes(self):
        source = (PROJECT_ROOT / "packaging" / "repair_gpu_env.bat").read_text(encoding="utf-8")
        build_environment = source.index("call :prepare_build_environment")
        preflight = source.index("repair_gpu_env.py\" --preflight-repair")
        active_change = source.index('"venv\\Scripts\\python.exe" -m pip install')
        self.assertLess(build_environment, preflight)
        self.assertLess(preflight, active_change)
        self.assertNotIn('call :prepare_build_environment >>', source)
        self.assertIn('"--preflight-only"', source)
        self.assertNotIn(r'del /s /q "detectron2\detectron2\*.pyd"', source.lower())
        self.assertIn("VsDevCmd.bat", source)

    def test_repair_only_cleans_temporary_detectron2_artifacts(self):
        source = (PROJECT_ROOT / "packaging" / "repair_gpu_env.bat").read_text(encoding="utf-8")
        self.assertIn(r'%TEMP_REPAIR_DIR%\detectron2\build', source)
        self.assertNotIn('if exist "detectron2\\build" rmdir', source.lower())

    def test_repair_uses_legacy_build_tooling_and_matching_compatibility_pins(self):
        source = (PROJECT_ROOT / "packaging" / "repair_gpu_env.bat").read_text(encoding="utf-8")
        self.assertIn("setuptools==65.5.0", source)
        self.assertIn("set \"COMPAT_PACKAGES=numpy==1.25.2 Pillow==8.4.0", source)
        self.assertIn('set "DISTUTILS_USE_SDK=1"', source)
        self.assertIn(r'VC\Tools\MSVC\14.3*', source)
        temp_pins = source.index('"%TEMP_PYTHON%" -m pip install --force-reinstall %COMPAT_PACKAGES%')
        wheel_build = source.index('"%TEMP_PYTHON%" -m pip wheel')
        self.assertLess(temp_pins, wheel_build)

    def test_pyinstaller_spec_collects_native_ml_runtime_without_upx(self):
        source = (PROJECT_ROOT / "packaging" / "PolyVision.spec").read_text(encoding="utf-8")
        self.assertIn("collect_dynamic_libs", source)
        self.assertIn("collect_submodules", source)
        self.assertIn('"retraining_runtime"', source)
        self.assertIn('"detectron2"', source)
        self.assertIn('"torchvision.ops"', source)
        self.assertIn("detectron2._C", source)
        self.assertIn("upx=False", source)

    def test_build_script_gates_source_and_packaged_gpu_diagnostics(self):
        source = (PROJECT_ROOT / "packaging" / "build_exe.bat").read_text(encoding="utf-8")
        source_gate = source.index(
            "venv\\Scripts\\python.exe UI\\PolyVisionMain.py --diagnose-retraining --require-gpu --json"
        )
        build_command = source.index('pyinstaller --clean --noconfirm "%~dp0PolyVision.spec"')
        packaged_gate = source.index(
            "dist\\PolyVision\\PolyVision.exe --diagnose-retraining --require-gpu --json"
        )
        success_message = source.index("Build and packaged GPU diagnostics completed successfully.")
        self.assertLess(source_gate, build_command)
        self.assertLess(build_command, packaged_gate)
        self.assertLess(packaged_gate, success_message)
        self.assertIn("goto fail", source)


class RepairPreflightTests(unittest.TestCase):
    def test_cuda_11_8_msvc_version_range(self):
        version_text, version = repair_preflight._normalize_msvc_version(
            "Microsoft (R) C/C++ Optimizing Compiler Version 19.39.33523 for x64"
        )
        self.assertEqual(version_text, "19.39")
        self.assertEqual(version, (19, 39))
        self.assertLess(version, repair_preflight.MAXIMUM_MSVC_VERSION_EXCLUSIVE)

        _, incompatible = repair_preflight._normalize_msvc_version(
            "Microsoft (R) C/C++ Optimizing Compiler Version 19.44.35219 for x64"
        )
        self.assertGreaterEqual(incompatible, repair_preflight.MAXIMUM_MSVC_VERSION_EXCLUSIVE)

    def test_missing_current_torch_is_replaceable_not_a_preflight_error(self):
        with patch.dict(sys.modules, {"torch": None}):
            check, error = repair_preflight._current_torch_cuda_check()
        self.assertIn("will be replaced", check)
        self.assertEqual(error, "")


if __name__ == "__main__":
    unittest.main()

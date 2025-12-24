#!/usr/bin/env python3
"""
Universal GPU Detection (NeurInSpectre)

This module is used by the `neurinspectre gpu ...` CLI commands.

Why this lives inside the package:
- Console entrypoints (e.g., `neurinspectre`) do **not** reliably import repo-root
  helper scripts (like `universal_gpu_detector.py`) because the working directory
  is not guaranteed to be on `sys.path`.
- Packaging it here ensures `neurinspectre gpu detect` works from any directory
  in an installed environment.
"""

import json
import platform
import subprocess
import tempfile
import time
from pathlib import Path


class UniversalGPUDetector:
    """Detect GPU hardware and framework acceleration across common platforms."""

    def __init__(self) -> None:
        self.system_info = {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        }
        self.gpu_info = {
            "apple_silicon": {},
            "nvidia": {},
            "amd": {},
            "intel": {},
        }
        self.last_report = None

    def detect_apple_silicon(self) -> bool:
        """Detect Apple Silicon GPU (MPS)."""
        print("🍎 Apple Silicon GPU Detection:")
        print("=" * 35)

        if platform.machine() != "arm64" or platform.system() != "Darwin":
            print("❌ Not an Apple Silicon system")
            return False

        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                displays = data.get("SPDisplaysDataType", [])
                for display in displays:
                    chipset = display.get("sppci_model", "Unknown")
                    vram = display.get("spdisplays_vram", "Unknown")
                    if (
                        "Apple" in str(chipset)
                        or "M1" in str(chipset)
                        or "M2" in str(chipset)
                        or "M3" in str(chipset)
                    ):
                        print(f"✅ Apple GPU: {chipset}")
                        if vram != "Unknown":
                            print(f"   VRAM: {vram}")
                        self.gpu_info["apple_silicon"] = {
                            "chipset": chipset,
                            "vram": vram,
                            "detected": True,
                        }
                        break
        except Exception:
            # Non-fatal: keep going (MPS test below is more important anyway)
            pass

        # Check PyTorch MPS
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                print("✅ PyTorch MPS: Available")
                print("   Metal Performance Shaders: Ready")

                # Lightweight sanity test
                device = torch.device("mps")
                _ = torch.randn(10, 10, device=device)
                print("   MPS Device Test: Passed")

                self.gpu_info["apple_silicon"].setdefault("detected", True)
                self.gpu_info["apple_silicon"]["pytorch_mps"] = True
                return True

            print("❌ PyTorch MPS: Not available")
        except ImportError:
            print("❌ PyTorch: Not installed")
        except Exception as e:
            print(f"❌ PyTorch MPS: Error - {e}")

        return bool(self.gpu_info["apple_silicon"].get("detected"))

    def detect_nvidia(self) -> bool:
        """Detect NVIDIA GPUs via nvidia-smi and check CUDA framework support."""
        print("\n⚡ NVIDIA GPU Detection:")
        print("=" * 25)

        try:
            result = subprocess.run(["nvidia-smi", "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                print("❌ NVIDIA SMI: Not available")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ NVIDIA SMI: Not found")
            return False

        print("✅ NVIDIA SMI: Available")

        gpus = []
        try:
            gpu_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if gpu_result.returncode == 0:
                for line in gpu_result.stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        gpu_info = {"name": parts[0], "memory_mb": int(parts[1]), "driver_version": parts[2]}
                        gpus.append(gpu_info)
                        print(f"   GPU: {gpu_info['name']} ({gpu_info['memory_mb']} MB)")
        except Exception:
            pass

        self.gpu_info["nvidia"] = {"gpus": gpus, "detected": True}
        self.check_cuda_frameworks()
        return True

    def check_cuda_frameworks(self) -> None:
        """Check CUDA support in common frameworks."""
        print("   🔥 CUDA Framework Support:")

        # PyTorch CUDA
        try:
            import torch

            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                print(f"      ✅ PyTorch CUDA: {device_count} device(s)")
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    print(f"         GPU {i}: {props.name}")
            else:
                print("      ❌ PyTorch CUDA: Not available")
        except ImportError:
            print("      ❌ PyTorch: Not installed")

        # TensorFlow GPU
        try:
            import tensorflow as tf

            gpus = tf.config.experimental.list_physical_devices("GPU")
            if gpus:
                print(f"      ✅ TensorFlow GPU: {len(gpus)} device(s)")
            else:
                print("      ❌ TensorFlow GPU: No devices")
        except ImportError:
            print("      ❌ TensorFlow: Not installed")
        except Exception:
            print("      ⚠️ TensorFlow GPU check failed")

    def detect_amd(self) -> bool:
        """Detect AMD GPUs (ROCm)."""
        print("\n🔴 AMD GPU Detection:")
        print("=" * 20)

        try:
            result = subprocess.run(["rocm-smi", "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✅ ROCm SMI: Available")
                self.gpu_info["amd"]["detected"] = True
                return True
            print("❌ ROCm SMI: Not available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ ROCm SMI: Not found")

        # Best-effort ROCm check via torch
        try:
            import torch

            hip = getattr(torch.version, "hip", None)
            if hip and torch.cuda.is_available():
                print("✅ PyTorch ROCm: Available")
                self.gpu_info["amd"]["detected"] = True
                return True
        except Exception:
            pass

        print("❌ No AMD GPU support detected")
        return False

    def detect_intel(self) -> bool:
        """Detect Intel GPUs (best-effort)."""
        print("\n🔵 Intel GPU Detection:")
        print("=" * 22)

        if platform.system() == "Linux":
            try:
                result = subprocess.run(["lspci", "-nn"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    intel_gpus = [ln.strip() for ln in result.stdout.split("\n") if "VGA" in ln and "Intel" in ln]
                    if intel_gpus:
                        print(f"✅ Intel GPUs detected: {len(intel_gpus)}")
                        for gpu in intel_gpus[:5]:
                            print(f"   {gpu}")
                        self.gpu_info["intel"]["detected"] = True
                        return True
            except Exception:
                pass

        elif platform.system() == "Darwin":
            try:
                result = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and "Intel" in result.stdout:
                    print("✅ Intel GPU detected in system")
                    self.gpu_info["intel"]["detected"] = True
                    return True
            except Exception:
                pass

        print("❌ No Intel GPU detected")
        return False

    def get_gpu_memory_usage(self) -> None:
        """Best-effort GPU memory usage reporting."""
        print("\n📊 GPU Memory Usage:")
        print("=" * 20)

        if self.gpu_info["nvidia"].get("detected"):
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        if not line.strip():
                            continue
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 3:
                            name = parts[0]
                            used = int(parts[1])
                            total = int(parts[2])
                            usage_percent = (used / total) * 100.0 if total else 0.0
                            print(f"   ⚡ {name}: {used}/{total} MB ({usage_percent:.1f}%)")
            except Exception:
                pass

        if self.gpu_info["apple_silicon"].get("detected"):
            print("   🍎 Apple Silicon: Unified Memory Architecture")
            print("      (GPU shares system memory)")

    def detect_running_models(self) -> None:
        """Detect likely AI/ML processes (best-effort)."""
        print("\n🧠 Running AI Models:")
        print("=" * 20)

        ai_processes = []
        try:
            import psutil  # type: ignore

            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    cmdline = " ".join(proc.info.get("cmdline") or [])
                    ai_keywords = [
                        "torch",
                        "tensorflow",
                        "keras",
                        "transformers",
                        "train",
                        "model",
                        "gpu",
                        "cuda",
                        "mps",
                    ]
                    if any(k in cmdline.lower() for k in ai_keywords):
                        ai_processes.append(
                            {
                                "pid": proc.info.get("pid"),
                                "name": proc.info.get("name"),
                                "command": (cmdline[:100] + "...") if len(cmdline) > 100 else cmdline,
                            }
                        )
                except Exception:
                    continue
        except ImportError:
            print("   ❌ psutil not available for process detection")
            return

        if ai_processes:
            print(f"   Found {len(ai_processes)} potential AI processes:")
            for p in ai_processes[:5]:
                print(f"      PID {p['pid']}: {p['name']}")
                print(f"         {p['command']}")
        else:
            print("   No AI processes currently detected")

    def generate_summary(self) -> dict:
        """Return a compact summary dict."""
        print("\n🎯 GPU Detection Summary:")
        print("=" * 30)

        detected_gpus = []
        if self.gpu_info["apple_silicon"].get("detected"):
            detected_gpus.append("Apple Silicon (MPS)")
        if self.gpu_info["nvidia"].get("detected"):
            nvidia_count = len(self.gpu_info["nvidia"].get("gpus", []))
            detected_gpus.append(f"NVIDIA ({nvidia_count} GPU(s))")
        if self.gpu_info["amd"].get("detected"):
            detected_gpus.append("AMD (ROCm)")
        if self.gpu_info["intel"].get("detected"):
            detected_gpus.append("Intel")

        if detected_gpus:
            print(f"✅ Detected GPU Types: {', '.join(detected_gpus)}")
        else:
            print("❌ No GPU acceleration detected")

        frameworks = []
        try:
            import torch

            if torch.cuda.is_available():
                frameworks.append("PyTorch CUDA")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                frameworks.append("PyTorch MPS")
        except Exception:
            pass

        try:
            import tensorflow as tf

            if tf.config.experimental.list_physical_devices("GPU"):
                frameworks.append("TensorFlow GPU")
        except Exception:
            pass

        if frameworks:
            print(f"🔧 AI Framework Support: {', '.join(frameworks)}")
        else:
            print("❌ No GPU-accelerated AI frameworks detected")

        return {"detected_gpus": detected_gpus, "supported_frameworks": frameworks, "gpu_info": self.gpu_info}

    def save_report(self) -> None:
        """Write a JSON report to CWD (or temp dir if CWD is not writable)."""
        summary = self.generate_summary()
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": self.system_info,
            "gpu_detection": self.gpu_info,
            "summary": summary,
        }
        self.last_report = report_data

        try:
            out = Path("universal_gpu_report.json")
            out.write_text(json.dumps(report_data, indent=2, default=str))
            print(f"\n💾 Detailed report saved to: {out}")
        except Exception:
            try:
                fallback = Path(tempfile.gettempdir()) / "universal_gpu_report.json"
                fallback.write_text(json.dumps(report_data, indent=2, default=str))
                print(f"\n💾 Detailed report saved to: {fallback}")
            except Exception:
                pass

    def run_full_detection(self) -> dict:
        """Run complete GPU detection across all platforms."""
        print("🌍 Universal GPU Detection")
        print("=" * 30)
        print(f"System: {self.system_info['platform']}")
        print(f"Architecture: {self.system_info['machine']}")
        print(f"Python: {self.system_info['python_version']}")

        # Detect all GPU types
        self.detect_apple_silicon()
        self.detect_nvidia()
        self.detect_amd()
        self.detect_intel()

        self.get_gpu_memory_usage()
        self.detect_running_models()

        summary = self.generate_summary()

        # Non-fatal save
        try:
            self.save_report()
        except Exception:
            pass

        return summary


def main() -> None:
    detector = UniversalGPUDetector()
    summary = detector.run_full_detection()
    print("\n🎉 Detection Complete!")
    if summary.get("detected_gpus"):
        print(f"   GPU Types: {', '.join(summary['detected_gpus'])}")
        print(f"   Frameworks: {', '.join(summary['supported_frameworks'])}")
    else:
        print("   No GPU acceleration available")


if __name__ == "__main__":
    main()


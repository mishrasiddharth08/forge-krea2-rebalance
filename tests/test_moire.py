import importlib.util
import inspect
import sys
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


modules = types.ModuleType("modules")
modules.scripts = types.SimpleNamespace(ScriptBuiltinUI=object, AlwaysVisible=True)
modules.shared = types.SimpleNamespace(cmd_opts=types.SimpleNamespace(lora_dir=None, lora_dirs=[]))
sys.modules["modules"] = modules
ui = types.ModuleType("modules.ui_components")
ui.InputAccordion = object
sys.modules["modules.ui_components"] = ui

path = Path(__file__).with_name("krea2_rebalance.py")
spec = importlib.util.spec_from_file_location("krea_moire_test", path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class MoireTests(unittest.TestCase):
    def test_kernel_and_default(self):
        expected = np.array([-1, 6, -15, 20, -15, 6, -1], np.float32) / 64
        np.testing.assert_array_equal(m.MOIRE_NOTCH_KERNEL, expected)
        self.assertEqual(inspect.signature(m._apply_moire_filter).parameters["amount"].default, 0.35)

    def test_checkerboard_is_attenuated(self):
        checker = ((np.indices((32, 32)).sum(0) & 1) * 255).astype(np.uint8)
        rgb = np.repeat(checker[..., None], 3, axis=2)
        out = np.asarray(m._apply_moire_filter(Image.fromarray(rgb, "RGB"), 1.0))
        self.assertLess(out.astype(float).std(), rgb.astype(float).std() * 0.05)

    def test_constant_preserved_at_boundaries(self):
        rgb = np.full((5, 6, 3), 137, np.uint8)
        out = np.asarray(m._apply_moire_filter(Image.fromarray(rgb, "RGB"), 1.0))
        np.testing.assert_array_equal(out, rgb)

    def test_zero_amount_has_exact_pixels(self):
        image = Image.fromarray(np.arange(63, dtype=np.uint8).reshape(7, 3, 3), "RGB")
        out = m._apply_moire_filter(image, 0)
        self.assertIs(out, image)
        np.testing.assert_array_equal(np.asarray(out), np.asarray(image))

    def test_alpha_and_metadata_preserved(self):
        rgba = np.zeros((8, 8, 4), np.uint8)
        rgba[..., :3] = np.indices((8, 8)).sum(0)[..., None] % 2 * 255
        rgba[..., 3] = np.arange(8, dtype=np.uint8)[:, None] * 31
        image = Image.fromarray(rgba, "RGBA")
        image.info["fixture"] = "kept"
        out = m._apply_moire_filter(image, 1)
        np.testing.assert_array_equal(np.asarray(out)[..., 3], rgba[..., 3])
        self.assertEqual(out.info["fixture"], "kept")

    def test_hook_is_off_by_default_and_uses_appended_args(self):
        script = m.Krea2RebalanceScript()
        image = Image.fromarray(np.tile([[0, 255], [255, 0]], (8, 8))[..., None].repeat(3, 2).astype(np.uint8), "RGB")
        p = types.SimpleNamespace(extra_generation_params={})
        pp = types.SimpleNamespace(image=image)
        legacy = [None] * 13
        legacy[11], legacy[12] = "stripes", True
        script.postprocess_image(p, pp, *legacy)
        self.assertIs(pp.image, image)
        script.postprocess_image(p, pp, *([None] * 13), True, 1.0)
        self.assertIsNot(pp.image, image)
        self.assertIs(p.extra_generation_params["Qwen Moire Cleanup"], True)
        self.assertEqual(p.extra_generation_params["Qwen Moire Cleanup Strength"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

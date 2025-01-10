import parametrize_from_file as pff
from pathlib import Path

COVERAGE_CONFIG = Path(__file__).parents[1] / 'pyproject.toml'
REF_IMAGE_DIR = Path(__file__).parent / 'ref_screenshots'
INPUT_DIR = Path(__file__).parent / 'pymol_inputs'

def run_pymol(commands, img_path):
    import os
    from subprocess import run
    from itertools import repeat
    from more_itertools import interleave
    from pytest import skip

    pymol = [
            os.environ.get('MMVOX_PYTEST_PYMOL', 'pymol'),
            '-ck',
            '-d', 'import coverage',
            '-d', 'coverage.process_startup()',
            '-d', 'import macromol_voxelize.pymol',
            *interleave(repeat('-d'), commands),
            '-g', str(img_path.resolve()),
    ]
    env = {
            **os.environ,
            'COVERAGE_PROCESS_START': str(COVERAGE_CONFIG),
    }

    try:
        run(pymol, env=env, check=True)
    except FileNotFoundError:
        skip("PyMOL not found")

def compare_images(expected, actual, *, tol):
    from matplotlib.testing.compare import compare_images
    from pytest import fail

    if not expected.exists():
        fail(f"Reference image not found: {expected}\nTest image: {actual}\nIf the test image looks right, rename it to the above reference path and rerun the test.")

    else:
        diff = compare_images(expected, actual, tol)
        if diff is not None:
            fail(diff)


@pff.parametrize(
        schema=[
            pff.cast(tol=float),
            pff.defaults(tol=0.1),
        ],
)
def test_pymol_screenshot(commands, tol, request, tmp_path):
    commands = [
            x.replace('$IN', str(INPUT_DIR.resolve()))
            for x in commands
    ]
    img_path = tmp_path / f'{request.node.callspec.id}.png'

    run_pymol(commands, img_path=img_path)

    compare_images(
            expected=REF_IMAGE_DIR / img_path.name,
            actual=img_path,
            tol=tol,
    )


import os
import shutil
import sys
from pathlib import Path
from typing import Union

from tqdm import tqdm


def copytree_progress(
    src: Union[Path, str], dst: Union[Path, str], desc: str = None, recopy=False, symlinks=False, ignore=None, copy_function=shutil.copy2, ignore_dangling_symlinks=False, dirs_exist_ok=False
):
    """
    TODO it would be great to get rid of this
    """
    if isinstance(src, str):
        src = Path(src)

    if isinstance(dst, str):
        dst = Path(dst)

    total_files = sum([len(files) for _, _, files in os.walk(src)])

    prog = tqdm(total=total_files, unit="file(s)", desc=desc, file=sys.stdout)

    def copy_tick(s, d):
        if recopy or not os.path.exists(d):
            copy_function(s, d)

        prog.update()

    shutil.copytree(src, dst, symlinks, ignore, copy_tick, ignore_dangling_symlinks, dirs_exist_ok)

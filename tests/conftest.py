import sys, pathlib, pytest

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

def get_solver_cmd(bin_name="crash_solver"):
    root = ROOT_DIR
    local_bin = root / "bin" / bin_name
    if not local_bin.exists():
        build_bin = root / "build" / "bin" / bin_name
        if build_bin.exists():
            local_bin = build_bin

    if sys.platform == "win32":
        drive = root.drive[0].lower()
        rel_posix = root.as_posix()[3:]
        wsl_path = f"/mnt/{drive}/{rel_posix}/bin/{bin_name}"
        return ["wsl", wsl_path]
    else:
        return [str(local_bin)]

def to_solver_path(p: pathlib.Path) -> str:
    p_abs = p.resolve()
    if sys.platform == "win32":
        drive = p_abs.drive[0].lower()
        rel = p_abs.as_posix()[3:]
        return f"/mnt/{drive}/{rel}"
    return str(p_abs)

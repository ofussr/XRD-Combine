"""Per-user Windows file associations for compiled XRD Combine builds."""

from __future__ import annotations

from pathlib import Path
import sys


ASSOCIATION_SUFFIXES = (".xrdml", ".raw", ".xy", ".cif")
PROG_ID = "XRDCombine.File"
CLASSES_ROOT = r"Software\Classes"


def open_command(executable: str | Path) -> str:
    """Build the quoted shell command used by the Windows registry."""

    return f'"{Path(executable)}" "%1"'


def runtime_executable() -> Path | None:
    """Return the current compiled Windows executable, never python.exe."""

    if sys.platform != "win32":
        return None
    compiled = bool(getattr(sys, "frozen", False)) or "__compiled__" in globals()
    if not compiled:
        return None
    executable = Path(sys.executable).resolve()
    return executable if executable.suffix.lower() == ".exe" else None


class WindowsFileAssociations:
    """Register and remove only the associations owned by XRD Combine."""

    def __init__(self, executable: str | Path | None = None) -> None:
        self.executable = (
            Path(executable).resolve() if executable is not None else runtime_executable()
        )

    @property
    def platform_supported(self) -> bool:
        return sys.platform == "win32"

    @property
    def available(self) -> bool:
        return self.platform_supported and self.executable is not None

    @property
    def application_key(self) -> str:
        name = self.executable.name if self.executable is not None else "XRD Combine.exe"
        return rf"{CLASSES_ROOT}\Applications\{name}"

    @property
    def prog_id_key(self) -> str:
        return rf"{CLASSES_ROOT}\{PROG_ID}"

    def _registry(self):
        if not self.platform_supported:
            raise OSError("Windows file associations are unavailable on this platform.")
        import winreg

        return winreg

    def _require_executable(self) -> Path:
        if self.executable is None:
            raise OSError("File associations are available only in a compiled build.")
        return self.executable

    def _set_value(self, path: str, name: str, value: str) -> None:
        registry = self._registry()
        with registry.CreateKeyEx(
            registry.HKEY_CURRENT_USER,
            path,
            0,
            registry.KEY_WRITE,
        ) as key:
            registry.SetValueEx(key, name, 0, registry.REG_SZ, value)

    def _set_empty_value(self, path: str, name: str) -> None:
        registry = self._registry()
        with registry.CreateKeyEx(
            registry.HKEY_CURRENT_USER,
            path,
            0,
            registry.KEY_WRITE,
        ) as key:
            registry.SetValueEx(key, name, 0, registry.REG_NONE, b"")

    def _has_value(self, path: str, name: str) -> bool:
        registry = self._registry()
        try:
            with registry.OpenKey(
                registry.HKEY_CURRENT_USER,
                path,
                0,
                registry.KEY_READ,
            ) as key:
                registry.QueryValueEx(key, name)
        except (FileNotFoundError, OSError):
            return False
        return True

    def _default_value(self, path: str) -> str | None:
        registry = self._registry()
        try:
            with registry.OpenKey(
                registry.HKEY_CURRENT_USER,
                path,
                0,
                registry.KEY_READ,
            ) as key:
                value, _kind = registry.QueryValueEx(key, "")
        except (FileNotFoundError, OSError):
            return None
        return str(value)

    def _delete_value(self, path: str, name: str) -> None:
        registry = self._registry()
        try:
            with registry.OpenKey(
                registry.HKEY_CURRENT_USER,
                path,
                0,
                registry.KEY_SET_VALUE,
            ) as key:
                registry.DeleteValue(key, name)
        except (FileNotFoundError, OSError):
            pass

    def _delete_tree(self, path: str) -> None:
        registry = self._registry()
        try:
            with registry.OpenKey(
                registry.HKEY_CURRENT_USER,
                path,
                0,
                registry.KEY_READ | registry.KEY_WRITE,
            ) as key:
                children: list[str] = []
                index = 0
                while True:
                    try:
                        children.append(registry.EnumKey(key, index))
                    except OSError:
                        break
                    index += 1
        except (FileNotFoundError, OSError):
            return
        for child in children:
            self._delete_tree(path + "\\" + child)
        try:
            registry.DeleteKey(registry.HKEY_CURRENT_USER, path)
        except (FileNotFoundError, OSError):
            pass

    def _notify_shell(self) -> None:
        try:
            from ctypes import windll

            windll.shell32.SHChangeNotify(0x08000000, 0x0000, None, None)
        except (AttributeError, OSError):
            pass

    def register(self) -> None:
        executable = self._require_executable()
        command = open_command(executable)
        icon = f'"{executable}",0'

        self._set_value(self.application_key, "FriendlyAppName", "XRD Combine")
        self._set_value(
            self.application_key + r"\shell\open\command",
            "",
            command,
        )
        self._set_value(self.prog_id_key, "", "XRD Combine file")
        self._set_value(self.prog_id_key + r"\DefaultIcon", "", icon)
        self._set_value(self.prog_id_key + r"\shell\open\command", "", command)

        for suffix in ASSOCIATION_SUFFIXES:
            self._set_value(
                self.application_key + r"\SupportedTypes",
                suffix,
                "",
            )
            self._set_empty_value(
                rf"{CLASSES_ROOT}\{suffix}\OpenWithProgids",
                PROG_ID,
            )
        self._notify_shell()

    def unregister(self) -> None:
        self._registry()
        for suffix in ASSOCIATION_SUFFIXES:
            self._delete_value(
                rf"{CLASSES_ROOT}\{suffix}\OpenWithProgids",
                PROG_ID,
            )
        self._delete_tree(self.application_key)
        self._delete_tree(self.prog_id_key)
        self._notify_shell()

    def status(self) -> dict[str, bool]:
        if not self.available:
            return {suffix: False for suffix in ASSOCIATION_SUFFIXES}
        expected_command = open_command(self._require_executable())
        command_matches = (
            self._default_value(self.application_key + r"\shell\open\command")
            == expected_command
            and self._default_value(self.prog_id_key + r"\shell\open\command")
            == expected_command
        )
        return {
            suffix: command_matches
            and self._has_value(
                self.application_key + r"\SupportedTypes",
                suffix,
            )
            and self._has_value(
                rf"{CLASSES_ROOT}\{suffix}\OpenWithProgids",
                PROG_ID,
            )
            for suffix in ASSOCIATION_SUFFIXES
        }


__all__ = [
    "ASSOCIATION_SUFFIXES",
    "PROG_ID",
    "WindowsFileAssociations",
    "open_command",
    "runtime_executable",
]

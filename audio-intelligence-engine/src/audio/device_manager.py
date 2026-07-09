"""Utilities for discovering and selecting audio input devices."""

from __future__ import annotations

from typing import List, Optional

import sounddevice as sd


class AudioDeviceManager:
    """Simple wrapper around sounddevice input device discovery."""

    def list_input_devices(self) -> List[dict]:
        """Return a list of available input devices."""
        devices = sd.query_devices()
        if isinstance(devices, dict):
            devices = [devices]
        return [
            {
                "index": idx,
                "name": device.get("name", f"Device {idx}"),
                "max_input_channels": device.get("max_input_channels", 0),
            }
            for idx, device in enumerate(devices)
            if isinstance(device, dict) and device.get("max_input_channels", 0) > 0
        ]

    def has_input_devices(self) -> bool:
        return bool(self.list_input_devices())

    def choose_input_device(self, preferred_name: Optional[str] = None) -> Optional[int]:
        """Choose an input device by name when possible, otherwise use the first one."""
        devices = self.list_input_devices()
        if not devices:
            return None

        if preferred_name:
            for device in devices:
                if preferred_name.lower() in device["name"].lower():
                    return device["index"]

        return devices[0]["index"]

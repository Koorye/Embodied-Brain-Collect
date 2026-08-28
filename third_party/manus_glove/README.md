# manus_glove

Python-only CFFI wrapper for ManusSDK.

# Usage
```bash
pip install "manus_glove @ git+https://github.com/etaoxing/manus_glove.git"

# run glove visualization
python -m manus_glove.run --debug
```

The SDK library is resolved automatically per platform. When this package is
vendored alongside `MANUS_Core_3.1.1_SDK/` (as in this repo's `third_party/`),
the vendored binary is loaded directly:
- `MANUS_Core_3.1.1_SDK/ManusSDK_v3.1.1/SDKMinimalClient_Linux/ManusSDK/lib/libManusSDK_Integrated.so`
- `MANUS_Core_3.1.1_SDK/ManusSDK_v3.1.1/SDKMinimalClient_Windows/ManusSDK/lib/ManusSDK.dll`

Otherwise it falls back to `~/.cache/manus_glove/lib`, downloading from the
Manus installer on first use. Windows additionally requires the Microsoft
Visual C++ Redistributable.

(Optional) put calibration files in `~/.cache/manus_glove/`.

## Linux: install udev rules
```bash
sudo tee /etc/udev/rules.d/70-manus-hid.rules << 'EOF'
# HIDAPI/libusb
SUBSYSTEMS=="usb", ATTRS{idVendor}=="3325", MODE:="0666"
SUBSYSTEMS=="usb", ATTRS{idVendor}=="1915", ATTRS{idProduct}=="83fd", MODE:="0666"
# HIDAPI/hidraw
KERNEL=="hidraw*", ATTRS{idVendor}=="3325", MODE:="0666"
EOF

# reload udev
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Windows needs no udev rules; the SDK talks to the dongle / MANUS Core directly
through the vendor drivers.

References:
- [`tetra-python-sdk/tetra/manus.py`](https://github.com/tetra-dynamics/tetra-python-sdk/blob/main/tetra/manus.py)
- [`wuji-technology/wuji-teleop-ros2`](https://github.com/wuji-technology/wuji-teleop-ros2)
- [`Wonikrobotics-git/allegro_hand_teleoperation`](https://github.com/Wonikrobotics-git/allegro_hand_teleoperation)

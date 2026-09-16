"""Python CFFI wrapper for ManusSDK, mirroring ManusDataPublisher.cpp logic.

Usage:
    from manus_glove import ManusDataPublisher

    with ManusDataPublisher() as pub:
        data = pub.GetGloveData(glove_id)
"""

import copy
import logging
import os
import sys
import threading
import time

import cffi

from ._cdef import CDEF
from ._enums import (
    AxisPolarity,
    AxisView,
    ChainTypeToString,
    ConnectionType,
    ErgonomicsDataType,
    ErgonomicsDataTypeToSide,
    ErgonomicsDataTypeToString,
    HandMotion,
    JointTypeToString,
    SDKReturnCode,
    SetGloveCalibrationReturnCode,
    Side,
    SideToString,
)
from .sdk import resolve_lib_path

logger = logging.getLogger(__name__)


class ManusSDKError(Exception):
    """Raised when an SDK call returns a non-success code."""

    def __init__(self, func_name: str, code: int):
        self.func_name = func_name
        self.code = code
        try:
            code_name = SDKReturnCode(code).name
        except ValueError:
            code_name = str(code)
        super().__init__(f"{func_name} failed: {code_name} ({code})")


class ManusDataPublisher:
    """CFFI-based Python wrapper for the ManusSDK, mirroring the C++ ManusDataPublisher class.

    Provides the same callback-driven data collection and thread-safe access pattern
    as the C++ implementation, without ROS2 dependencies.
    """

    # Mirrors static ManusDataPublisher::s_Instance
    _s_Instance = None

    def __init__(
        self,
        lib_path: str | None = None,
        connection_type: ConnectionType = ConnectionType.Integrated,
        ip: str = "",
        world_space: bool = True,
        hand_motion: HandMotion = HandMotion.NoMotion,
        calibration_dir: str | None = None,
        left_calibration_file: str | None = "LeftMetaglovePro.mcal",
        right_calibration_file: str | None = "RightMetaglovePro.mcal",
        debug: bool = False,
    ):
        if ManusDataPublisher._s_Instance is not None:
            raise RuntimeError("ManusDataPublisher can only be initialized once.")
        ManusDataPublisher._s_Instance = self

        # udev rules grant HID access on Linux only; Windows needs no such step.
        if sys.platform.startswith("linux"):
            _UDEV_RULES = "/etc/udev/rules.d/70-manus-hid.rules"
            if not os.path.isfile(_UDEV_RULES):
                raise RuntimeError(f"udev rules not found: {_UDEV_RULES}")

        # Debug logging
        self.debug = debug
        logger.setLevel(logging.DEBUG if debug else logging.WARNING)

        # CFFI setup
        self._ffi = cffi.FFI()
        self._ffi.cdef(CDEF)

        if lib_path is None:
            lib_path = resolve_lib_path()
        self._lib = self._ffi.dlopen(lib_path)

        # Connection settings (mirrors C++ member variables)
        self.m_ConnectionType = connection_type
        self.m_Ip = ip
        self.m_WorldSpace = world_space
        self.m_HandMotion = hand_motion

        # Calibration file paths and load-state flags (mirrors C++ m_LeftCalibrationLoaded etc.)
        if calibration_dir is not None:
            _cal_dir = calibration_dir
        else:
            _cal_dir = os.path.join(os.path.expanduser("~"), ".cache", "manus_glove")

        def _cal_path(filename: str | None) -> str | None:
            if not filename:
                return None
            p = os.path.join(_cal_dir, filename)
            return p if os.path.isfile(p) else None

        self.m_LeftCalibrationPath = _cal_path(left_calibration_file)
        self.m_RightCalibrationPath = _cal_path(right_calibration_file)
        self.m_LeftCalibrationLoaded = False
        self.m_RightCalibrationLoaded = False

        # Coordinate system (mirrors C++ defaults: XFromViewer, PositiveZ, Right, 1.0)
        self.m_CoordinateSystem = {
            "view": AxisView.XFromViewer,
            "up": AxisPolarity.PositiveZ,
            "handedness": Side.Right,
            "unitScale": 1.0,
        }

        # Data stores (mirrors C++ std::map<uint32_t, ...>)
        self.m_RawSkeletonMutex = threading.Lock()
        self.m_GloveDataMap: dict[int, dict] = {}

        self.m_RawSensorDataMutex = threading.Lock()
        self.m_RawSensorDataMap: dict[int, dict] = {}

        self.m_ErgonomicsMutex = threading.Lock()
        self.m_ErgonomicsDataMap: dict[int, list[float]] = {}

        self.m_LandscapeMutex = threading.Lock()
        self.m_NewLandscape: dict | None = None
        self.m_Landscape: dict | None = None

        # Node info cache (mirrors C++ m_NodeInfo)
        self.m_NodeInfo: dict[int, list[dict]] | None = None
        self._m_NodeInfoGloveId: int | None = None

        # Keep callback references alive (prevent GC)
        self._callbacks = {}

    # ------------------------------------------------------------------
    # Lifecycle (mirrors C++ constructor flow)
    # ------------------------------------------------------------------

    def Initialize(self) -> None:
        """Initialize the SDK (mirrors C++ ManusDataPublisher::Initialize)."""
        self.InitializeSDK()

    def InitializeSDK(self) -> None:
        """Initialize SDK, register callbacks, set coordinate system.

        Mirrors C++ ManusDataPublisher::InitializeSDK.
        """
        if self.m_ConnectionType in (ConnectionType.Invalid,):
            raise ManusSDKError("InitializeSDK", SDKReturnCode.InvalidArgument)

        # Register log callback before Initialize so startup messages are captured.
        # When debug=False the callback is a no-op, suppressing all SDK output.
        @self._ffi.callback("void(int, const char *, uint32_t)")
        def _OnLogCallback(level, message, length):
            if self.debug:
                msg = self._ffi.string(message, length).decode("utf-8", errors="replace")
                logger.debug("[SDK] %s", msg.rstrip())

        self._callbacks["log"] = _OnLogCallback
        self._lib.CoreSdk_RegisterCallbackForOnLog(_OnLogCallback)

        if self.m_ConnectionType == ConnectionType.Integrated:
            rc = self._lib.CoreSdk_InitializeIntegrated()
        else:
            rc = self._lib.CoreSdk_InitializeCore()

        if rc != SDKReturnCode.Success:
            raise ManusSDKError("CoreSdk_Initialize", rc)

        self.RegisterAllCallbacks()

        coord = self._ffi.new("CoordinateSystemVUH *")
        self._lib.CoordinateSystemVUH_Init(coord)
        coord.view = self.m_CoordinateSystem["view"]
        coord.up = self.m_CoordinateSystem["up"]
        coord.handedness = self.m_CoordinateSystem["handedness"]
        coord.unitScale = self.m_CoordinateSystem["unitScale"]

        rc = self._lib.CoreSdk_InitializeCoordinateSystemWithVUH(coord[0], self.m_WorldSpace)
        if rc != SDKReturnCode.Success:
            raise ManusSDKError("CoreSdk_InitializeCoordinateSystemWithVUH", rc)

    def RegisterAllCallbacks(self) -> None:
        """Register all SDK stream callbacks.

        Mirrors C++ ManusDataPublisher::RegisterAllCallbacks.
        """

        # Raw skeleton stream
        @self._ffi.callback("void(const SkeletonStreamInfo *)")
        def _OnRawSkeletonStreamCallback(p_RawSkeletonStreamInfo):
            self._OnRawSkeletonStreamCallback(p_RawSkeletonStreamInfo)

        self._callbacks["raw_skeleton"] = _OnRawSkeletonStreamCallback
        rc = self._lib.CoreSdk_RegisterCallbackForRawSkeletonStream(_OnRawSkeletonStreamCallback)
        if rc != SDKReturnCode.Success:
            raise ManusSDKError("CoreSdk_RegisterCallbackForRawSkeletonStream", rc)

        # Raw device data stream
        @self._ffi.callback("void(const RawDeviceDataInfo *)")
        def _OnRawDeviceDataStreamCallback(p_RawDeviceDataInfo):
            self._OnRawDeviceDataStreamCallback(p_RawDeviceDataInfo)

        self._callbacks["raw_device"] = _OnRawDeviceDataStreamCallback
        rc = self._lib.CoreSdk_RegisterCallbackForRawDeviceDataStream(_OnRawDeviceDataStreamCallback)
        if rc != SDKReturnCode.Success:
            raise ManusSDKError("CoreSdk_RegisterCallbackForRawDeviceDataStream", rc)

        # Ergonomics stream
        @self._ffi.callback("void(const ErgonomicsStream *)")
        def _OnErgonomicsStreamCallback(p_ErgonomicsStream):
            self._OnErgonomicsStreamCallback(p_ErgonomicsStream)

        self._callbacks["ergonomics"] = _OnErgonomicsStreamCallback
        rc = self._lib.CoreSdk_RegisterCallbackForErgonomicsStream(_OnErgonomicsStreamCallback)
        if rc != SDKReturnCode.Success:
            raise ManusSDKError("CoreSdk_RegisterCallbackForErgonomicsStream", rc)

        # Landscape stream
        @self._ffi.callback("void(const Landscape *)")
        def _OnLandscapeCallback(p_Landscape):
            self._OnLandscapeCallback(p_Landscape)

        self._callbacks["landscape"] = _OnLandscapeCallback
        rc = self._lib.CoreSdk_RegisterCallbackForLandscapeStream(_OnLandscapeCallback)
        if rc != SDKReturnCode.Success:
            raise ManusSDKError("CoreSdk_RegisterCallbackForLandscapeStream", rc)

    def Connect(self, retry: bool = True, retry_interval: float = 1.0) -> None:
        """Connect to Manus Core host.

        Mirrors C++ ManusDataPublisher::Connect with optional retry loop
        matching the C++ constructor's while-loop pattern.
        """
        while True:
            result = self._TryConnect()
            if result:
                break
            if not retry:
                raise ManusSDKError("Connect", SDKReturnCode.NotConnected)
            logger.info("Could not connect, trying again in %.1fs.", retry_interval)
            time.sleep(retry_interval)

        # Set hand motion mode (mirrors C++ constructor)
        rc = self._lib.CoreSdk_SetRawSkeletonHandMotion(int(self.m_HandMotion))
        if rc != SDKReturnCode.Success:
            logger.error("Failed to set hand motion mode: %d", rc)

        logger.info("Manus Core connected.")

    def _TryConnect(self) -> bool:
        """Single connection attempt. Returns True on success.

        Mirrors C++ ManusDataPublisher::Connect.
        """
        rc = self._lib.CoreSdk_LookForHosts(5, False)
        if rc != SDKReturnCode.Success:
            return False

        p_count = self._ffi.new("uint32_t *")
        rc = self._lib.CoreSdk_GetNumberOfAvailableHostsFound(p_count)
        if rc != SDKReturnCode.Success or p_count[0] == 0:
            return False

        count = p_count[0]
        hosts = self._ffi.new("ManusHost[]", count)
        rc = self._lib.CoreSdk_GetAvailableHostsFound(hosts, count)
        if rc != SDKReturnCode.Success:
            return False

        auto_connect = not self.m_Ip
        host_selection = 0

        if auto_connect:
            logger.info("Autoconnecting to the first host found.")
        else:
            logger.info("Looking for host with IP address: %s", self.m_Ip)
            for i in range(count):
                ip_str = self._ffi.string(hosts[i].ipAddress).decode("utf-8")
                host_ip = ip_str.split(":")[0]
                if host_ip == self.m_Ip:
                    host_selection = i
                    break

        rc = self._lib.CoreSdk_ConnectToHost(hosts[host_selection])
        if rc == SDKReturnCode.NotConnected:
            return False

        return True

    def ShutDown(self) -> None:
        """Shut down the SDK and clean up.

        Mirrors C++ ManusDataPublisher::~ManusDataPublisher + ShutDown.
        """
        # Zero out vibration on all haptic gloves (mirrors C++ destructor)
        with self.m_LandscapeMutex:
            landscape = self.m_Landscape

        if landscape is not None:
            for glove in landscape.get("gloves", []):
                if glove.get("isHaptics", False):
                    logger.info("Sending zero vibration to glove %d", glove["id"])
                    powers = self._ffi.new("float[5]", [0, 0, 0, 0, 0])
                    self._lib.CoreSdk_VibrateFingersForGlove(glove["id"], powers)

        rc = self._lib.CoreSdk_ShutDown()
        if rc != SDKReturnCode.Success:
            logger.error("CoreSdk_ShutDown failed: %d", rc)

        self.m_NodeInfo = None
        self._m_NodeInfoGloveId = None
        ManusDataPublisher._s_Instance = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.Initialize()
        self.Connect()
        return self

    def __exit__(self, *args):
        self.ShutDown()

    # ------------------------------------------------------------------
    # SDK Callbacks (mirrors C++ static callback methods)
    # ------------------------------------------------------------------

    def _OnRawSkeletonStreamCallback(self, p_RawSkeletonStreamInfo) -> None:
        """Mirrors C++ ManusDataPublisher::OnRawSkeletonStreamCallback."""
        info = self._ffi.new("RawSkeletonInfo *")

        with self.m_RawSkeletonMutex:
            for i in range(p_RawSkeletonStreamInfo.skeletonsCount):
                rc = self._lib.CoreSdk_GetRawSkeletonInfo(i, info)
                if rc != SDKReturnCode.Success:
                    if rc == SDKReturnCode.SdkNotAvailable:
                        return
                    logger.error("CoreSdk_GetRawSkeletonInfo error: %d", rc)
                    continue

                nodes_count = info.nodesCount
                if nodes_count == 0:
                    continue

                nodes = self._ffi.new("SkeletonNode[]", nodes_count)
                rc = self._lib.CoreSdk_GetRawSkeletonData(i, nodes, nodes_count)
                if rc != SDKReturnCode.Success:
                    if rc == SDKReturnCode.SdkNotAvailable:
                        return
                    logger.error("CoreSdk_GetRawSkeletonData error: %d", rc)
                    continue

                # Store as Python data (mirrors C++ ClientRawSkeleton)
                skeleton = {
                    "gloveId": info.gloveId,
                    "nodesCount": nodes_count,
                    "publishTime": info.publishTime.time,
                    "nodes": [],
                }
                for j in range(nodes_count):
                    n = nodes[j]
                    skeleton["nodes"].append(
                        {
                            "id": n.id,
                            "position": (
                                n.transform.position.x,
                                n.transform.position.y,
                                n.transform.position.z,
                            ),
                            "rotation": (
                                n.transform.rotation.x,
                                n.transform.rotation.y,
                                n.transform.rotation.z,
                                n.transform.rotation.w,
                            ),
                        }
                    )

                self.m_GloveDataMap[info.gloveId] = skeleton

    def _OnRawDeviceDataStreamCallback(self, p_RawDeviceDataInfo) -> None:
        """Mirrors C++ ManusDataPublisher::OnRawDeviceDataStreamCallback."""
        raw_data = self._ffi.new("RawDeviceData *")

        with self.m_RawSensorDataMutex:
            for i in range(p_RawDeviceDataInfo.rawDeviceDataCount):
                rc = self._lib.CoreSdk_GetRawDeviceData(i, raw_data)
                if rc != SDKReturnCode.Success:
                    continue

                sensor_data = {
                    "id": raw_data.id,
                    "sensorCount": raw_data.sensorCount,
                    "rotation": (
                        raw_data.rotation.x,
                        raw_data.rotation.y,
                        raw_data.rotation.z,
                        raw_data.rotation.w,
                    ),
                    "sensors": [],
                }
                for j in range(raw_data.sensorCount):
                    s = raw_data.sensorData[j]
                    sensor_data["sensors"].append(
                        {
                            "position": (
                                s.position.x,
                                s.position.y,
                                s.position.z,
                            ),
                            "rotation": (
                                s.rotation.x,
                                s.rotation.y,
                                s.rotation.z,
                                s.rotation.w,
                            ),
                        }
                    )

                self.m_RawSensorDataMap[raw_data.id] = sensor_data

    def _OnErgonomicsStreamCallback(self, p_ErgonomicsStream) -> None:
        """Mirrors C++ ManusDataPublisher::OnErgonomicsStreamCallback."""
        for i in range(p_ErgonomicsStream.dataCount):
            ergo = p_ErgonomicsStream.data[i]
            if ergo.isUserID:
                continue

            data = [ergo.data[j] for j in range(ErgonomicsDataType.MAX_SIZE)]

            with self.m_ErgonomicsMutex:
                self.m_ErgonomicsDataMap[ergo.id] = data

    def _OnLandscapeCallback(self, p_Landscape) -> None:
        """Mirrors C++ ManusDataPublisher::OnLandscapeCallback."""
        # Deep copy landscape data into Python dict
        landscape = {
            "dongleCount": p_Landscape.gloveDevices.dongleCount,
            "gloveCount": p_Landscape.gloveDevices.gloveCount,
            "gloves": [],
            "license": {
                "sdk": p_Landscape.settings.license.sdk,
                "integrated": p_Landscape.settings.license.integrated,
            },
        }

        for i in range(p_Landscape.gloveDevices.gloveCount):
            g = p_Landscape.gloveDevices.gloves[i]
            landscape["gloves"].append(
                {
                    "id": g.id,
                    "side": g.side,
                    "isHaptics": g.isHaptics,
                    "familyType": g.familyType,
                    "batteryPercentage": g.batteryPercentage,
                }
            )

        with self.m_LandscapeMutex:
            self.m_NewLandscape = landscape

    # ------------------------------------------------------------------
    # Data access (polling getters, replaces C++ PublishCallback)
    # ------------------------------------------------------------------

    def GetGloveIds(self) -> list[int]:
        """Return list of known glove IDs from the latest skeleton data."""
        with self.m_RawSkeletonMutex:
            return list(self.m_GloveDataMap.keys())

    def GetNodeInfo(self, glove_id: int) -> list[dict] | None:
        """Get node hierarchy info for a glove. Cached after first retrieval.

        Mirrors C++ m_NodeInfo retrieval in PublishCallback.
        Returns list of dicts with keys: nodeId, parentId, chainType, side, fingerJointType.
        """
        if self.m_NodeInfo is not None and self._m_NodeInfoGloveId == glove_id:
            return self.m_NodeInfo

        # Get nodes count from skeleton data
        with self.m_RawSkeletonMutex:
            skel = self.m_GloveDataMap.get(glove_id)

        if skel is None:
            return None

        nodes_count = skel["nodesCount"]
        node_info_arr = self._ffi.new("NodeInfo[]", nodes_count)
        rc = self._lib.CoreSdk_GetRawSkeletonNodeInfoArray(glove_id, node_info_arr, nodes_count)
        if rc != SDKReturnCode.Success:
            logger.error("CoreSdk_GetRawSkeletonNodeInfoArray error: %d", rc)
            return None

        result = []
        for i in range(nodes_count):
            ni = node_info_arr[i]
            result.append(
                {
                    "nodeId": ni.nodeId,
                    "parentId": ni.parentId,
                    "chainType": ni.chainType,
                    "side": ni.side,
                    "fingerJointType": ni.fingerJointType,
                    "chainTypeStr": ChainTypeToString(ni.chainType),
                    "jointTypeStr": JointTypeToString(ni.fingerJointType),
                }
            )

        self.m_NodeInfo = result
        self._m_NodeInfoGloveId = glove_id
        return result

    def GetGloveData(self, glove_id: int) -> dict | None:
        """Get a snapshot of all data for a glove.

        Mirrors the per-glove message construction in C++ PublishCallback.
        Returns dict with keys: glove_id, side, raw_nodes, ergonomics, raw_sensors.
        """
        # Update landscape (mirrors PublishCallback landscape swap)
        with self.m_LandscapeMutex:
            if self.m_NewLandscape is not None:
                self.m_Landscape = self.m_NewLandscape
                self.m_NewLandscape = None

        if self.m_Landscape is None:
            return None

        # Find this glove in landscape
        glove_info = None
        for g in self.m_Landscape.get("gloves", []):
            if g["id"] == glove_id:
                glove_info = g
                break

        if glove_info is None:
            return None

        # Copy skeleton data
        with self.m_RawSkeletonMutex:
            skel = self.m_GloveDataMap.get(glove_id)

        if skel is None or skel["nodesCount"] == 0:
            return None

        # Get node info (lazy init, mirrors C++ m_NodeInfo pattern)
        node_info = self.GetNodeInfo(glove_id)

        # Build raw nodes (mirrors C++ raw_nodes message construction)
        raw_nodes = []
        for node in skel["nodes"]:
            node_entry = {
                "id": node["id"],
                "position": node["position"],
                "rotation": node["rotation"],
            }
            if node_info is not None and node["id"] < len(node_info):
                ni = node_info[node["id"]]
                node_entry["parentId"] = ni["parentId"]
                node_entry["jointType"] = ni["jointTypeStr"]
                node_entry["chainType"] = ni["chainTypeStr"]
            raw_nodes.append(node_entry)

        # Ergonomics (mirrors C++ ergonomics message construction)
        with self.m_ErgonomicsMutex:
            ergo_data = self.m_ErgonomicsDataMap.get(glove_id)

        ergonomics = []
        if ergo_data is not None:
            side_val = glove_info["side"]
            for y in range(ErgonomicsDataType.MAX_SIZE):
                if ErgonomicsDataTypeToSide(y) != side_val:
                    continue
                ergonomics.append(
                    {
                        "type": ErgonomicsDataTypeToString(y),
                        "value": ergo_data[y],
                    }
                )

        # Raw sensor data (mirrors C++ raw sensor message construction)
        with self.m_RawSensorDataMutex:
            raw_sensor = self.m_RawSensorDataMap.get(glove_id)

        raw_sensors = None
        if raw_sensor is not None and raw_sensor["sensorCount"] > 0:
            raw_sensors = {
                "orientation": raw_sensor["rotation"],
                "sensors": raw_sensor["sensors"],
            }

        return {
            "glove_id": glove_id,
            "side": SideToString(glove_info["side"]),
            "raw_node_count": skel["nodesCount"],
            "raw_nodes": raw_nodes,
            "ergonomics": ergonomics,
            "raw_sensors": raw_sensors,
        }

    def GetLandscape(self) -> dict | None:
        """Get current landscape snapshot.

        Returns glove list with id, side, haptics status.
        """
        with self.m_LandscapeMutex:
            if self.m_NewLandscape is not None:
                self.m_Landscape = self.m_NewLandscape
                self.m_NewLandscape = None

        if self.m_Landscape is None:
            return None

        return copy.deepcopy(self.m_Landscape)

    # ------------------------------------------------------------------
    # Haptics (mirrors C++ vibration support)
    # ------------------------------------------------------------------

    def VibrateFingersForGlove(self, glove_id: int, powers: list[float]) -> None:
        """Send vibration command to a haptic glove.

        Mirrors C++ ManusDataPublisher::OnVibrationCommand.
        powers: list of 5 floats (Thumb, Index, Middle, Ring, Pinky), each 0.0-1.0.
        """
        # Verify glove supports haptics (mirrors C++ check)
        with self.m_LandscapeMutex:
            landscape = self.m_Landscape

        if landscape is not None:
            glove_found = False
            for g in landscape.get("gloves", []):
                if g["id"] == glove_id:
                    if not g.get("isHaptics", False):
                        return
                    glove_found = True
                    break
            if not glove_found:
                return

        # Clamp to 5 elements, 0-1 range (mirrors C++ clamping)
        intensities = [0.0] * 5
        for i in range(min(len(powers), 5)):
            intensities[i] = max(0.0, min(1.0, powers[i]))

        c_powers = self._ffi.new("float[5]", intensities)
        rc = self._lib.CoreSdk_VibrateFingersForGlove(glove_id, c_powers)
        if rc != SDKReturnCode.Success:
            logger.error("Failed to vibrate glove %d: SDK error %d", glove_id, rc)

    # ------------------------------------------------------------------
    # Calibration (mirrors C++ ManusDataPublisher::LoadCalibrationFile)
    # ------------------------------------------------------------------

    def LoadCalibrationFile(self, glove_id: int, side: int, calibration_path: str) -> bool:
        """Load a .mcal calibration file and send it to the SDK for a given glove.

        Mirrors C++ ManusDataPublisher::LoadCalibrationFile.
        Returns True on success, False if the file is missing or the SDK call fails.
        """
        if not os.path.isfile(calibration_path):
            logger.warning("Calibration file not found: %s", calibration_path)
            return False

        with open(calibration_path, "rb") as f:
            data = f.read()

        length = len(data)
        c_data = self._ffi.new("unsigned char[]", data)
        p_result = self._ffi.new("int *")

        rc = self._lib.CoreSdk_SetGloveCalibration(glove_id, c_data, length, p_result)
        if rc != SDKReturnCode.Success:
            logger.error("CoreSdk_SetGloveCalibration SDK error for glove %d: %d", glove_id, rc)
            return False

        cal_rc = p_result[0]
        if cal_rc == SetGloveCalibrationReturnCode.Success:
            side_str = SideToString(side)
            logger.info("Calibration loaded successfully for %s glove (ID: %d)", side_str, glove_id)
            return True
        else:
            logger.error("Failed to load calibration for glove ID %d, error code: %d", glove_id, cal_rc)
            return False

    def LoadCalibrationFiles(self) -> None:
        """Load calibration files for all currently known gloves using the paths
        supplied at construction time (left_calibration_path / right_calibration_path).

        Mirrors the auto-load logic in C++ ManusDataPublisher::PublishCallback.
        Call this after Connect() once the landscape is available.
        """
        landscape = self.GetLandscape()
        if landscape is None:
            logger.warning("LoadCalibrationFiles: landscape not yet available.")
            return

        for glove in landscape.get("gloves", []):
            glove_id = glove["id"]
            side = glove["side"]

            if side == Side.Left and not self.m_LeftCalibrationLoaded:
                if self.m_LeftCalibrationPath is None:
                    logger.warning("No left calibration path configured.")
                elif self.LoadCalibrationFile(glove_id, side, self.m_LeftCalibrationPath):
                    self.m_LeftCalibrationLoaded = True

            elif side == Side.Right and not self.m_RightCalibrationLoaded:
                if self.m_RightCalibrationPath is None:
                    logger.warning("No right calibration path configured.")
                elif self.LoadCalibrationFile(glove_id, side, self.m_RightCalibrationPath):
                    self.m_RightCalibrationLoaded = True

    # ------------------------------------------------------------------
    # String helpers (mirrors C++ helper methods, delegated to _enums.py)
    # ------------------------------------------------------------------

    @staticmethod
    def SideToString(side: int) -> str:
        return SideToString(side)

    @staticmethod
    def JointTypeToString(joint_type: int) -> str:
        return JointTypeToString(joint_type)

    @staticmethod
    def ChainTypeToString(chain_type: int) -> str:
        return ChainTypeToString(chain_type)

    @staticmethod
    def ErgonomicsDataTypeToSide(ergo_type: int) -> Side:
        return ErgonomicsDataTypeToSide(ergo_type)

    @staticmethod
    def ErgonomicsDataTypeToString(ergo_type: int) -> str:
        return ErgonomicsDataTypeToString(ergo_type)

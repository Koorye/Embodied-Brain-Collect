"""Python enums mirroring ManusSDKTypes.h C enums."""

from enum import IntEnum


class SDKReturnCode(IntEnum):
    Success = 0
    Error = 1
    InvalidArgument = 2
    ArgumentSizeMismatch = 3
    UnsupportedStringSizeEncountered = 4
    SdkNotAvailable = 5
    HostFinderNotAvailable = 6
    DataNotAvailable = 7
    MemoryError_ = 8
    InternalError = 9
    FunctionCalledAtWrongTime = 10
    NotConnected = 11
    ConnectionTimeout = 12
    InvalidID = 13
    NullPointer = 14
    InvalidSequence = 15
    NoCoordinateSystemSet = 16
    SdkIsTerminating = 17
    StubNullPointer = 18
    SkeletonNotLoaded = 19
    FunctionNotAvailable = 20


class ConnectionType(IntEnum):
    Invalid = 0
    Integrated = 1
    Local = 2
    Remote = 3


class Side(IntEnum):
    Invalid = 0
    Left = 1
    Right = 2
    Center = 3


class ChainType(IntEnum):
    Invalid = 0
    Arm = 1
    Leg = 2
    Neck = 3
    Spine = 4
    FingerThumb = 5
    FingerIndex = 6
    FingerMiddle = 7
    FingerRing = 8
    FingerPinky = 9
    Pelvis = 10
    Head = 11
    Shoulder = 12
    Hand = 13
    Foot = 14
    Toe = 15


class FingerJointType(IntEnum):
    Invalid = 0
    Metacarpal = 1
    Proximal = 2
    Intermediate = 3
    Distal = 4
    Tip = 5


class HandMotion(IntEnum):
    NoMotion = 0
    IMU = 1
    Tracker = 2
    Tracker_RotationOnly = 3
    Auto = 4


class AxisView(IntEnum):
    Invalid = 0
    ZFromViewer = 1
    YFromViewer = 2
    XFromViewer = 3
    XToViewer = 4
    YToViewer = 5
    ZToViewer = 6


class AxisPolarity(IntEnum):
    Invalid = 0
    NegativeZ = 1
    NegativeY = 2
    NegativeX = 3
    PositiveX = 4
    PositiveY = 5
    PositiveZ = 6


class ErgonomicsDataType(IntEnum):
    LeftFingerThumbMCPSpread = 0
    LeftFingerThumbMCPStretch = 1
    LeftFingerThumbPIPStretch = 2
    LeftFingerThumbDIPStretch = 3
    LeftFingerIndexMCPSpread = 4
    LeftFingerIndexMCPStretch = 5
    LeftFingerIndexPIPStretch = 6
    LeftFingerIndexDIPStretch = 7
    LeftFingerMiddleMCPSpread = 8
    LeftFingerMiddleMCPStretch = 9
    LeftFingerMiddlePIPStretch = 10
    LeftFingerMiddleDIPStretch = 11
    LeftFingerRingMCPSpread = 12
    LeftFingerRingMCPStretch = 13
    LeftFingerRingPIPStretch = 14
    LeftFingerRingDIPStretch = 15
    LeftFingerPinkyMCPSpread = 16
    LeftFingerPinkyMCPStretch = 17
    LeftFingerPinkyPIPStretch = 18
    LeftFingerPinkyDIPStretch = 19
    RightFingerThumbMCPSpread = 20
    RightFingerThumbMCPStretch = 21
    RightFingerThumbPIPStretch = 22
    RightFingerThumbDIPStretch = 23
    RightFingerIndexMCPSpread = 24
    RightFingerIndexMCPStretch = 25
    RightFingerIndexPIPStretch = 26
    RightFingerIndexDIPStretch = 27
    RightFingerMiddleMCPSpread = 28
    RightFingerMiddleMCPStretch = 29
    RightFingerMiddlePIPStretch = 30
    RightFingerMiddleDIPStretch = 31
    RightFingerRingMCPSpread = 32
    RightFingerRingMCPStretch = 33
    RightFingerRingPIPStretch = 34
    RightFingerRingDIPStretch = 35
    RightFingerPinkyMCPSpread = 36
    RightFingerPinkyMCPStretch = 37
    RightFingerPinkyPIPStretch = 38
    RightFingerPinkyDIPStretch = 39
    MAX_SIZE = 40


class SetGloveCalibrationReturnCode(IntEnum):
    Error = 0
    Success = 1
    VersionError = 2
    WrongSideError = 3
    GloveNotFoundError = 4
    UserServiceError = 5
    DeserializationError = 6


# String conversion helpers matching C++ ManusDataPublisher methods

_SIDE_STRINGS = {
    Side.Left: "Left",
    Side.Right: "Right",
}

_JOINT_TYPE_STRINGS = {
    FingerJointType.Metacarpal: "MCP",
    FingerJointType.Proximal: "PIP",
    FingerJointType.Intermediate: "IP",
    FingerJointType.Distal: "DIP",
    FingerJointType.Tip: "TIP",
}

_CHAIN_TYPE_STRINGS = {
    ChainType.Arm: "Arm",
    ChainType.Leg: "Leg",
    ChainType.Neck: "Neck",
    ChainType.Spine: "Spine",
    ChainType.FingerThumb: "Thumb",
    ChainType.FingerIndex: "Index",
    ChainType.FingerMiddle: "Middle",
    ChainType.FingerRing: "Ring",
    ChainType.FingerPinky: "Pinky",
    ChainType.Pelvis: "Pelvis",
    ChainType.Head: "Head",
    ChainType.Shoulder: "Shoulder",
    ChainType.Hand: "Hand",
    ChainType.Foot: "Foot",
    ChainType.Toe: "Toe",
}

# Maps each ErgonomicsDataType to its display name (side-agnostic, matching C++)
_ERGO_TYPE_STRINGS = {
    0: "ThumbMCPSpread",
    1: "ThumbMCPStretch",
    2: "ThumbPIPStretch",
    3: "ThumbDIPStretch",
    4: "IndexSpread",
    5: "IndexMCPStretch",
    6: "IndexPIPStretch",
    7: "IndexDIPStretch",
    8: "MiddleSpread",
    9: "MiddleMCPStretch",
    10: "MiddlePIPStretch",
    11: "MiddleDIPStretch",
    12: "RingSpread",
    13: "RingMCPStretch",
    14: "RingPIPStretch",
    15: "RingDIPStretch",
    16: "PinkySpread",
    17: "PinkyMCPStretch",
    18: "PinkyPIPStretch",
    19: "PinkyDIPStretch",
    # Right side maps to same strings
    20: "ThumbMCPSpread",
    21: "ThumbMCPStretch",
    22: "ThumbPIPStretch",
    23: "ThumbDIPStretch",
    24: "IndexSpread",
    25: "IndexMCPStretch",
    26: "IndexPIPStretch",
    27: "IndexDIPStretch",
    28: "MiddleSpread",
    29: "MiddleMCPStretch",
    30: "MiddlePIPStretch",
    31: "MiddleDIPStretch",
    32: "RingSpread",
    33: "RingMCPStretch",
    34: "RingPIPStretch",
    35: "RingDIPStretch",
    36: "PinkySpread",
    37: "PinkyMCPStretch",
    38: "PinkyPIPStretch",
    39: "PinkyDIPStretch",
}


def SideToString(side: int) -> str:
    return _SIDE_STRINGS.get(side, "Invalid")


def JointTypeToString(joint_type: int) -> str:
    return _JOINT_TYPE_STRINGS.get(joint_type, "Invalid")


def ChainTypeToString(chain_type: int) -> str:
    return _CHAIN_TYPE_STRINGS.get(chain_type, "Invalid")


def ErgonomicsDataTypeToSide(ergo_type: int) -> Side:
    if 0 <= ergo_type <= 19:
        return Side.Left
    elif 20 <= ergo_type <= 39:
        return Side.Right
    return Side.Invalid


def ErgonomicsDataTypeToString(ergo_type: int) -> str:
    return _ERGO_TYPE_STRINGS.get(ergo_type, "Invalid")

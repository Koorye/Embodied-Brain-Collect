"""CFFI C definitions string for ManusSDK types and functions.

Derived from ManusSDK/include/ManusSDKTypes.h and ManusSDK/include/ManusSDK.h.
All struct layouts must exactly match the C ABI for correct field offsets.
"""

CDEF = """

    /* ------------------------------------------------------------------ */
    /* Core types                                                         */
    /* ------------------------------------------------------------------ */

    typedef struct ManusVec3 {
        float x;
        float y;
        float z;
    } ManusVec3;

    typedef struct ManusQuaternion {
        float w;
        float x;
        float y;
        float z;
    } ManusQuaternion;

    typedef struct ManusTransform {
        ManusVec3 position;
        ManusQuaternion rotation;
        ManusVec3 scale;
    } ManusTransform;

    typedef struct ManusTimestamp {
        uint64_t time;
    } ManusTimestamp;

    typedef struct IMUCalibrationInfo {
        uint32_t mag;
        uint32_t acc;
        uint32_t gyr;
        uint32_t sys;
    } IMUCalibrationInfo;

    typedef struct Version {
        uint32_t major;
        uint32_t minor;
        uint32_t patch;
        char label[16];
        char sha[16];
        char tag[16];
    } Version;

    typedef struct FirmwareVersion {
        int32_t version;
        ManusTimestamp timestamp;
    } FirmwareVersion;

    typedef struct ManusVersion {
        char versionInfo[16];
    } ManusVersion;

    typedef struct ManusHost {
        char hostName[256];
        char ipAddress[40];
        Version manusCoreVersion;
    } ManusHost;

    typedef struct CoordinateSystemVUH {
        int view;       /* AxisView */
        int up;         /* AxisPolarity */
        int handedness; /* Side */
        float unitScale;
    } CoordinateSystemVUH;

    /* ------------------------------------------------------------------ */
    /* Skeleton stream                                                    */
    /* ------------------------------------------------------------------ */

    typedef struct SkeletonNode {
        uint32_t id;
        ManusTransform transform;
    } SkeletonNode;

    typedef struct RawSkeletonInfo {
        uint32_t gloveId;
        uint32_t nodesCount;
        ManusTimestamp publishTime;
    } RawSkeletonInfo;

    typedef struct SkeletonStreamInfo {
        ManusTimestamp publishTime;
        uint32_t skeletonsCount;
    } SkeletonStreamInfo;

    typedef struct NodeInfo {
        uint32_t nodeId;
        uint32_t parentId;
        int chainType;       /* ChainType */
        int side;            /* Side */
        int fingerJointType; /* FingerJointType */
    } NodeInfo;

    /* ------------------------------------------------------------------ */
    /* Raw device data stream                                             */
    /* ------------------------------------------------------------------ */

    typedef struct RawDeviceData {
        uint32_t id;
        uint32_t sensorCount;
        ManusTransform sensorData[5];
        ManusQuaternion rotation;
    } RawDeviceData;

    typedef struct RawDeviceDataInfo {
        ManusTimestamp publishTime;
        uint32_t rawDeviceDataCount;
    } RawDeviceDataInfo;

    /* ------------------------------------------------------------------ */
    /* Ergonomics stream                                                  */
    /* ------------------------------------------------------------------ */

    typedef struct ErgonomicsData {
        uint32_t id;
        bool isUserID;
        float data[40];  /* ErgonomicsDataType_MAX_SIZE */
    } ErgonomicsData;

    typedef struct ErgonomicsStream {
        ManusTimestamp publishTime;
        ErgonomicsData data[32]; /* MAX_NUMBER_OF_ERGONOMICS_DATA = MAX_NUMBER_OF_GLOVES = 32 */
        uint32_t dataCount;
    } ErgonomicsStream;

    /* ------------------------------------------------------------------ */
    /* Landscape (full struct hierarchy for correct field offsets)         */
    /* ------------------------------------------------------------------ */

    /* Device landscape */
    typedef struct DongleLandscapeData {
        uint32_t id;
        int classType;       /* DeviceClassType */
        int familyType;      /* DeviceFamilyType */
        bool isHaptics;

        Version hardwareVersion;
        Version firmwareVersion;
        ManusTimestamp firmwareTimestamp;

        uint32_t chargingState;
        int32_t channel;
        int updateStatus;    /* UpdateStatusEnum */

        char licenseType[64];
        ManusTimestamp lastSeen;

        uint32_t leftGloveID;
        uint32_t rightGloveID;

        int licenseLevel;    /* LicenseType */
        ManusTimestamp licenseExpiration;
        uint32_t licenseMaxNumberOfGlovePairs;

        uint32_t netDeviceID;
    } DongleLandscapeData;

    typedef struct GloveLandscapeData {
        uint32_t id;
        int classType;       /* DeviceClassType */
        int familyType;      /* DeviceFamilyType */
        int side;            /* Side */
        bool isHaptics;

        int pairedState;     /* DevicePairedState */
        uint32_t dongleID;

        Version hardwareVersion;
        Version firmwareVersion;
        ManusTimestamp firmwareTimestamp;

        int updateStatus;    /* UpdateStatusEnum */

        uint32_t batteryPercentage;
        int32_t transmissionStrength;

        IMUCalibrationInfo iMUCalibrationInfo[6]; /* MAX_NUM_IMUS_ON_GLOVE */

        ManusTimestamp lastSeen;
        bool excluded;

        uint32_t netDeviceID;
    } GloveLandscapeData;

    typedef struct DeviceLandscape {
        DongleLandscapeData dongles[16]; /* MAX_NUMBER_OF_DONGLES */
        uint32_t dongleCount;
        GloveLandscapeData gloves[32];   /* MAX_NUMBER_OF_GLOVES */
        uint32_t gloveCount;
    } DeviceLandscape;

    /* User landscape */
    typedef struct Measurement {
        int entryType;  /* MeasurementType */
        float value;
        int unit;       /* MeasurementUnit */
        int category;   /* MeasurementCategory */
        char displayName[64];
    } Measurement;

    typedef struct TrackerOffset {
        int entryType;  /* TrackerOffsetType */
        ManusVec3 translation;
        ManusQuaternion rotation;
    } TrackerOffset;

    typedef struct ExtraTrackerOffset {
        int entryType;  /* ExtraTrackerOffsetType */
        float value;
    } ExtraTrackerOffset;

    typedef struct UserProfileLandscapeData {
        int profileType;    /* ProfileType */
        Measurement measurements[20];          /* MeasurementType_MAX_SIZE */
        TrackerOffset trackerOffsets[16];       /* TrackerOffsetType_MAX_SIZE */
        ExtraTrackerOffset extraTrackerOffsets[4]; /* ExtraTrackerOffsetType_MAX_SIZE */
        ManusTimestamp leftGloveCalibrationTimestamp;
        ManusTimestamp rightGloveCalibrationTimestamp;
    } UserProfileLandscapeData;

    typedef struct UserLandscapeData {
        uint32_t id;
        char name[64];           /* MAX_NUM_CHARS_IN_USERNAME */
        uint32_t dongleID;
        uint32_t leftGloveID;
        uint32_t rightGloveID;
        UserProfileLandscapeData profile;
        uint32_t userIndex;
    } UserLandscapeData;

    typedef struct UserLandscape {
        UserLandscapeData users[16]; /* MAX_USERS */
        uint32_t userCount;
    } UserLandscape;

    /* Skeleton landscape */
    typedef struct SkeletonLandscapeData {
        uint32_t id;
        char session[256];       /* MAX_NUM_CHARS_IN_HOST_NAME */
        uint32_t userId;
        int type;                /* SkeletonType */
        char rootBoneName[256];  /* MAX_NUM_CHARS_IN_NODE_NAME */
        bool scaled;
    } SkeletonLandscapeData;

    typedef struct SkeletonLandscape {
        SkeletonLandscapeData skeletons[32]; /* MAX_NUMBER_OF_SKELETONS */
        uint32_t skeletonCount;
    } SkeletonLandscape;

    /* Tracker landscape */
    typedef struct TrackerLandscapeData {
        char id[32];             /* MAX_NUM_CHARS_IN_TRACKER_ID */
        int type;                /* TrackerType */
        int systemType;          /* TrackerSystemType */
        uint32_t user;
        bool isHMD;
        char manufacturer[32];   /* MAX_NUM_CHARS_IN_TRACKER_MANUFACTURER */
        char productName[32];    /* MAX_NUM_CHARS_IN_TRACKER_PRODUCTNAME */
    } TrackerLandscapeData;

    typedef struct TrackerLandscape {
        TrackerLandscapeData trackers[128]; /* MAX_NUMBER_OF_TRACKERS = 16*8 */
        uint32_t trackerCount;
    } TrackerLandscape;

    /* Settings landscape */
    typedef struct LicenseInfo {
        uint32_t maxGlovePairs;
        bool recording;
        bool exporting;
        bool advancedExporting;
        bool unitySession;
        bool unrealSession;
        bool openXRSession;
        bool sdk;
        bool raw;
        bool mobuSession;
        bool xsensSession;
        bool optitrackSession;
        bool qualisysSession;
        bool viconSession;
        bool nokovSession;
        bool icidoSession;
        bool siemensSession;
        bool vredSession;
        bool integrated;
        bool cortexSession;
        bool fzMotionSession;
        bool worldVizSession;
        bool noraxonSession;
        bool noitomSession;
        bool abletonSession;
    } LicenseInfo;

    typedef struct SettingsLandscape {
        Version manusCoreVersion;
        LicenseInfo license;
        bool playbackMode;
        bool ignoreSessionTimeOuts;
        FirmwareVersion firmwareOne;
        FirmwareVersion firmwareTwo;
        bool recordingMode;
        bool isNetDevice;
        bool isConnectedAsNetDevice;
    } SettingsLandscape;

    /* Net devices landscape */
    typedef struct NetDeviceLandscapeData {
        uint32_t netDeviceID;
        char hostname[256];     /* MAX_NUM_CHARS_IN_HOST_NAME */
        char ip[40];            /* MAX_NUM_CHARS_IN_IP_ADDRESS */
    } NetDeviceLandscapeData;

    typedef struct NetDevicesLandscape {
        uint32_t numberOfNetDevices;
        NetDeviceLandscapeData netDevices[16]; /* MAX_NUMBER_OF_NETDEVICES */
    } NetDevicesLandscape;

    /* Time landscape */
    typedef struct TimecodeInterface {
        char name[64];          /* MAX_NUM_CHARS_IN_TIMECODE_INTERFACE_STRINGS */
        char api[64];
        int index;
    } TimecodeInterface;

    typedef struct TimeLandscape {
        TimecodeInterface interfaces[32]; /* MAX_NUMBER_OF_AUDIO_INTERFACES */
        uint32_t interfaceCount;
        TimecodeInterface currentInterface;
        int fps;                /* TimecodeFPS */
        bool fakeTimecode;
        bool useSyncPulse;
        bool deviceKeepAlive;
        bool syncStatus;
        bool timecodeStatus;
        int32_t ltcChannel;
    } TimeLandscape;

    typedef struct GestureLandscapeData {
        uint32_t id;
        char name[64];          /* MAX_NUM_CHARS_IN_USERNAME */
    } GestureLandscapeData;

    /* Top-level Landscape */
    typedef struct Landscape {
        DeviceLandscape gloveDevices;
        UserLandscape users;
        SkeletonLandscape skeletons;
        TrackerLandscape trackers;
        SettingsLandscape settings;
        NetDevicesLandscape netDevices;
        TimeLandscape time;
        uint32_t gestureCount;
    } Landscape;

    /* ------------------------------------------------------------------ */
    /* Callback typedefs                                                  */
    /* ------------------------------------------------------------------ */

    typedef void (*ConnectedToCoreCallback_t)(const ManusHost *);
    typedef void (*DisconnectedFromCoreCallback_t)(const ManusHost *);
    typedef void (*LoggingCallback_t)(int, const char *, uint32_t);
    typedef void (*LandscapeStreamCallback_t)(const Landscape *);
    typedef void (*ErgonomicsStreamCallback_t)(const ErgonomicsStream *);
    typedef void (*RawSkeletonStreamCallback_t)(const SkeletonStreamInfo *);
    typedef void (*RawDeviceDataStreamCallback_t)(const RawDeviceDataInfo *);

    /* ------------------------------------------------------------------ */
    /* SDK functions                                                      */
    /* ------------------------------------------------------------------ */

    /* Lifecycle */
    int CoreSdk_InitializeIntegrated();
    int CoreSdk_InitializeCore();
    int CoreSdk_ShutDown();

    /* Connection */
    int CoreSdk_LookForHosts(uint32_t waitSeconds, bool loopbackOnly);
    int CoreSdk_GetNumberOfAvailableHostsFound(uint32_t *count);
    int CoreSdk_GetAvailableHostsFound(ManusHost *hosts, uint32_t count);
    int CoreSdk_ConnectToHost(ManusHost host);
    int CoreSdk_GetIsConnectedToCore(bool *connected);

    /* Coordinate system */
    int CoreSdk_InitializeCoordinateSystemWithVUH(CoordinateSystemVUH coordSystem, bool useWorldCoordinates);
    void CoordinateSystemVUH_Init(CoordinateSystemVUH *cs);

    /* Callbacks */
    int CoreSdk_RegisterCallbackForOnConnect(ConnectedToCoreCallback_t cb);
    int CoreSdk_RegisterCallbackForOnDisconnect(DisconnectedFromCoreCallback_t cb);
    int CoreSdk_RegisterCallbackForOnLog(LoggingCallback_t cb);
    int CoreSdk_RegisterCallbackForLandscapeStream(LandscapeStreamCallback_t cb);
    int CoreSdk_RegisterCallbackForErgonomicsStream(ErgonomicsStreamCallback_t cb);
    int CoreSdk_RegisterCallbackForRawSkeletonStream(RawSkeletonStreamCallback_t cb);
    int CoreSdk_RegisterCallbackForRawDeviceDataStream(RawDeviceDataStreamCallback_t cb);

    /* Raw skeleton data */
    int CoreSdk_GetRawSkeletonInfo(uint32_t skeletonIndex, RawSkeletonInfo *info);
    int CoreSdk_GetRawSkeletonData(uint32_t skeletonIndex, SkeletonNode *nodes, uint32_t nodeCount);
    int CoreSdk_GetRawSkeletonNodeInfoArray(uint32_t gloveId, NodeInfo *nodeInfoArray, uint32_t arraySize);

    /* Raw device data */
    int CoreSdk_GetRawDeviceData(uint32_t rawDeviceDataIndex, RawDeviceData *rawDeviceData);

    /* Hand motion */
    int CoreSdk_SetRawSkeletonHandMotion(int handMotion);
    int CoreSdk_GetRawSkeletonHandMotion(int *handMotion);

    /* Haptics */
    int CoreSdk_VibrateFingersForGlove(uint32_t gloveId, const float *powers);

    /* Gesture landscape */
    int CoreSdk_GetGestureLandscapeData(GestureLandscapeData *landscapeDataArray, uint32_t arraySize);

    /* Calibration */
    int CoreSdk_SetGloveCalibration(uint32_t gloveId, unsigned char *calibrationBytes, uint32_t bytesLength, int *result);

    /* Host init */
    void ManusHost_Init(ManusHost *host);
"""

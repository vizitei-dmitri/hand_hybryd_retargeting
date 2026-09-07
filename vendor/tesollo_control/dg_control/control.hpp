#ifndef DG_CONTROL
#define DG_CONTROL

#include <iostream>
#include <thread>
#include <chrono>
#include "DGSDK.h"
#include <stdlib.h>
#include <cstring>

#include <thread>
#include <mutex>
#include <functional>
#include <atomic>
#include <algorithm>
#include <cstdint>
#include <string>

#include <Eigen/Dense>

#include "../lockfree/lockfree.hpp"

namespace handcontrol
{
class DGControl
{
private:

    // ----------------- Singleton
    static DGControl* _instancePtr;
    DGControl(const char* ip = "169.254.186.72", int port = 502, int slaveID = 1);
    // ~DGControl();

    // ----------------- Callbacks
    std::atomic<bool> _g_connected{false};
    ReceivedGripperData _g_gripperData{};
    ReceivedFingertipSensorData _g_sensorData{};
    ReceivedGPIOData _g_gpioData{};
    DiagnosisSystem _g_diagnosisData{};
    std::atomic<int> _g_commPeriod{0};
    std::atomic<int> _g_processing{0};
    std::atomic<bool> _controlRunning{false};
    std::atomic<bool> _stopRequested{false};
    std::atomic<bool> _recoveryRequired{false};
    std::atomic<std::int64_t> _lastCommunicationNs{0};
    // Serialize explicit recovery and output; callbacks only update atomics/data.
    mutable std::mutex _motionMutex;
    std::atomic<bool> _systemStarted{false};
    std::atomic<bool> _temperatureSafe{false};
    std::atomic<bool> _servoKeepaliveEnabled{true};
    std::atomic<int> _lastMotionResult{DG_RESULT_NONE};
    std::atomic<bool> _everConnected{false};
    std::atomic<std::uint64_t> _disconnectCount{0};
    std::atomic<std::uint64_t> _reconnectCount{0};
    std::atomic<std::int64_t> _lastTelemetryNs{0};
    std::atomic<std::int64_t> _systemStartNs{0};
    std::atomic<bool> _latestCommandValid{false};

    std::atomic<int> _diagnosisProcess{0};
    std::atomic<int> _diagnosisStep{0};
    std::atomic<int> _diagnosisJointId{0};
    std::atomic<int> _diagnosisPeriod{0};
    std::atomic<int> _diagnosisJoint{0};
    std::atomic<int> _diagnosisTemperature{0};

    mutable std::mutex _gripperDataMutex;
    mutable std::mutex _commandMutex;

    // ----------------- Connections
    char _ip[MAX_GRIPPER_IP_ADDRESS_SIZE] = "169.254.186.72";
    int _port = 502;
    int _slaveID = 1;
    int _readTimeout = 1000;

    int _model = DG_MODEL_DG_5F_RIGHT;
    int _movingInpose = 1;
    int _type[MAX_RECEIVED_DATA_TYPE_COUNT] = {1,2,3,4,5,6};

    // ----------------- Dynamics
    float _P[MAX_JOINT_COUNT] = {1.0};
	float _D[MAX_JOINT_COUNT] = {2.0};

    // ----------------- Kinematics
    float _currentPos[MAX_JOINT_COUNT] = {0};
    float _currentCur[MAX_JOINT_COUNT] = {0};
    float _currentVel[MAX_JOINT_COUNT] = {0};
    float _currentTemp[MAX_JOINT_COUNT] = {0};

    const float _upperLimits[MAX_JOINT_COUNT] = {51, 0, 90, 90, 
                                                35, 115, 90, 90, 
                                                35, 112, 90, 90, 
                                                24, 109, 90, 90, 
                                                60, 35, 90, 90};

    const float _lowerLimits[MAX_JOINT_COUNT] = {-22, -180, -90, -90, 
                                                -24, -0, -90, -90, 
                                                -35, -0, -90, -90, 
                                                -35, -0, -90, -90, 
                                                -0, -24, -90, -90};
                                                
    const float _deltaLimits = 7.0;

    const float _tempLimit = 65.0;

    float _tempPos[MAX_JOINT_COUNT] = {0};
    float _targetPos[MAX_JOINT_COUNT] = {
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0
    };
    float _latestCommandPos[MAX_JOINT_COUNT] = {0};

    // ----------------- Queues
    ring_buffer<Eigen::Array<double,MAX_JOINT_COUNT,1>> _target_joint_buffer;

    ring_buffer<Eigen::Array<double,MAX_JOINT_COUNT,1>> _current_joint_buffer;
    ring_buffer<Eigen::Array<double,MAX_JOINT_COUNT,1>> _current_current_buffer;
    ring_buffer<Eigen::Array<double,MAX_JOINT_COUNT,1>> _current_velocity_buffer;
    ring_buffer<Eigen::Array<double,MAX_JOINT_COUNT,1>> _current_temperature_buffer;


    std::jthread _control;

    Eigen::Array<double,MAX_JOINT_COUNT,1> _msgTargetPos;

    Eigen::Array<double,MAX_JOINT_COUNT,1> _msgCurrentPos;
    Eigen::Array<double,MAX_JOINT_COUNT,1> _msgCurrentCur;
    Eigen::Array<double,MAX_JOINT_COUNT,1> _msgCurrentVel;
    Eigen::Array<double,MAX_JOINT_COUNT,1> _msgCurrentTemp;

    // ----------------- Callbacks
    static void _ConnectedCallback();
    static void _DisconnectedCallback();
    static void _ReceivedGripperDataCallback(ReceivedGripperData data);
    static void _CommunicationPeriodCallback(int period);
    static void _DiagnosisCallback(DiagnosisSystem diag);
    static void _FingertipCallback(ReceivedFingertipSensorData data);
    static void _GPIOCallback(ReceivedGPIOData data);
    static void _DataProcessingCallback(int status);

    void _setCallbacks();

    // ----------------- Loop
    void _loop();
    void _updatePos();

    bool _checkTemp();

public:

    DGControl(const DGControl& obj) = delete;
    DGControl& operator=(const DGControl& obj) = delete;

    static DGControl* getInstance(const char* ip = "169.254.186.72", int port = 502, int slaveID = 1);

    void start(bool servoKeepalive = true);
    void stop();

    bool setTragetPosition(const Eigen::Array<double,MAX_JOINT_COUNT,1> &t_position);
    bool getCurrentPosition(Eigen::Array<double,MAX_JOINT_COUNT,1> &c_position);
    bool getCurrentCurrent(Eigen::Array<double,MAX_JOINT_COUNT,1> &c_current);
    bool getCurrentVelocity(Eigen::Array<double,MAX_JOINT_COUNT,1> &c_velocity);
    bool getCurrentTemperature(Eigen::Array<double,MAX_JOINT_COUNT,1> &c_temperature);

    bool setTragetPosition(const float* t_position);
    bool getCurrentPosition(float* c_position);
    bool getCurrentCurrent(float* c_current);
    bool getCurrentVelocity(float* c_velocity);
    bool getCurrentTemperature(float* c_temperature);

    bool isConnected() const;
    bool isControlRunning() const;
    bool isSystemStarted() const;
    bool isTemperatureSafe() const;
    bool isServoKeepaliveEnabled() const;
    bool isMotionReady() const;
    bool isTelemetryValid() const;
    int getCommunicationRateHz() const;
    int getDataProcessingStatus() const;
    int getLastMotionResult() const;
    std::uint64_t getDisconnectCount() const;
    std::uint64_t getReconnectCount() const;
    int getDiagnosisProcess() const;
    int getDiagnosisStep() const;
    int getDiagnosisJointId() const;
    int getDiagnosisPeriod() const;
    int getDiagnosisJoint() const;
    int getDiagnosisTemperature() const;
    bool getLatestCommand(float* command) const;
    std::string motionReadyReason() const;
    double telemetryAgeMs() const;
    double communicationAgeMs() const;
    bool recoveryRequired() const { return _recoveryRequired.load(); }
    void recover(float* measuredPose, double timeoutSeconds = 2.0);
    void suspendMotion();
    bool getTelemetry(float* position, float* current, float* velocity, float* temperature) const;

};

void eigenArray2Array(const Eigen::Array<double,MAX_JOINT_COUNT,1> &eigen_array, float* array);
void array2EigenArray(float* array, Eigen::Array<double,MAX_JOINT_COUNT,1> &eigen_array);

int sign(float a, float threshold=0.001);

}

#endif

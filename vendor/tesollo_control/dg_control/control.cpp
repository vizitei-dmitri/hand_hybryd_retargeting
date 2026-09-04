#include "control.hpp"

using namespace handcontrol;

namespace
{
std::int64_t steadyNowNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}
}

DGControl *DGControl::_instancePtr = nullptr; 

DGControl::DGControl(const char* ip, int port, int slaveID):
_target_joint_buffer(16),
_current_joint_buffer(16),
_current_current_buffer(16),
_current_velocity_buffer(16),
_current_temperature_buffer(16)
{
    this->_port = port;
    this->_slaveID = slaveID;
    std::memcpy(this->_ip, ip, MAX_GRIPPER_IP_ADDRESS_SIZE);

    _msgTargetPos << 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
    _msgCurrentPos << 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
}

DGControl* DGControl::getInstance(const char* ip, int port, int slaveID)
{
    if (_instancePtr == nullptr) {
            _instancePtr = new DGControl(ip, port, slaveID);
    }
    return _instancePtr;
}

// --------------------------------------------------------------------------------

void DGControl::_ConnectedCallback()
{
    DGControl* control = getInstance();
    const bool wasConnected = control->_g_connected.exchange(true);
    if (!wasConnected && control->_everConnected.exchange(true))
    {
        control->_reconnectCount.fetch_add(1);
    }
}

void DGControl::_DisconnectedCallback()
{
    DGControl* control = getInstance();
    if (control->_g_connected.exchange(false))
    {
        control->_disconnectCount.fetch_add(1);
    }
    std::cerr << "DGSDK disconnected callback\n";
}

void DGControl::_ReceivedGripperDataCallback(ReceivedGripperData data)
{
    DGControl* control = getInstance();
    {
        std::lock_guard<std::mutex> lock(control->_gripperDataMutex);
        control->_g_gripperData = data;
    }
    control->_lastTelemetryNs.store(steadyNowNs());
}

void DGControl::_CommunicationPeriodCallback(int period)
{
    getInstance()->_g_commPeriod = period;
}

void DGControl::_DiagnosisCallback(DiagnosisSystem diag)
{
    DGControl* control = getInstance();
    control->_diagnosisProcess.store(diag.process);
    control->_diagnosisStep.store(diag.step);
    control->_diagnosisJointId.store(diag.jointId);
    control->_diagnosisPeriod.store(diag.period);
    control->_diagnosisJoint.store(diag.joint);
    control->_diagnosisTemperature.store(diag.temperature);
}

void DGControl::_FingertipCallback(ReceivedFingertipSensorData data)
{
    getInstance()->_g_sensorData = data;
}

void DGControl::_GPIOCallback(ReceivedGPIOData data)
{
    getInstance()->_g_gpioData = data;
}

void DGControl::_DataProcessingCallback(int status)
{
    getInstance()->_g_processing = status;
}


void DGControl::_setCallbacks()
{
    CallbackForOnConnected(DGControl::_ConnectedCallback);
    CallbackForOnDisconnected(DGControl::_DisconnectedCallback);
    CallbackForOnReceivedGripperData(DGControl::_ReceivedGripperDataCallback);
    CallbackForOnCommunicationPeriod(DGControl::_CommunicationPeriodCallback);
    CallbackForOnDiagnosisSystem(DGControl::_DiagnosisCallback);
    CallbackForOnReceivedFingertipSensorData(DGControl::_FingertipCallback);
    CallbackForOnReceivedGPIOData(DGControl::_GPIOCallback);
    CallbackForOnDataProcessing(DGControl::_DataProcessingCallback);
}

// --------------------------------------------------------------------------------

void DGControl::start(bool servoKeepalive)
{
    int success = 0;
    DG_RESULT result;

    GripperSystemSetting setting{};

    setting.communicationMode = COMMUNICATION_MODE_ETHERNET;
    setting.controlMode       = CONTROL_MODE_DEVELOPER;
    setting.port              = this->_port;
    setting.slaveID           = this->_slaveID;
    setting.readTimeout       = this->_readTimeout;
    std::memcpy(setting.ip, this->_ip, MAX_GRIPPER_IP_ADDRESS_SIZE);

    _lastTelemetryNs.store(0);
    _latestCommandValid.store(false);
    _systemStarted.store(false);

    result = SetGripperSystem(setting);
    std::cout << "SetGripperSystem: " << result << "\n";
    success += result;

    // --------------------------

    this->_setCallbacks();

    // --------------------------

    result = ConnectToGripper();
    std::cout << "ConnectToGripper: " << result << "\n";
    success += result;

    while (!_g_connected.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // --------------------------

    GripperSetting gs{};
    gs.model = DG_MODEL_DG_5F_RIGHT;
    gs.movingInpose = 1;
    int type[MAX_RECEIVED_DATA_TYPE_COUNT] = {1,2,3,4,5,6};
    std::memcpy(gs.receivedDataType, type, sizeof(type));

    result = SetGripperOption(gs);
    success += result;
    std::cout << "SetGripperOption: " << result << "\n";

    // --------------------------

    while (_g_commPeriod.load() < 200) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Capture the real pose before activating the servo system.  A DISARMED
    // hardware session will keep only this pose alive; it must never inherit
    // the firmware's default all-zero target.
    {
        std::lock_guard<std::mutex> lock(_gripperDataMutex);
        std::memcpy(_tempPos, _g_gripperData.joint, sizeof(_g_gripperData.joint));
    }
    _tempPos[16] = 0.0;
    _servoKeepaliveEnabled.store(servoKeepalive);
    _temperatureSafe.store(false);
    _lastMotionResult.store(DG_RESULT_NONE);

    result = SystemStart();
    std::cout << "SystemStart: " << result << "\n";
    success += result;
    _systemStarted.store(result == DG_RESULT_NONE);

    // Developer mode drops its control session if it receives no servo
    // traffic.  Immediately hold the measured pose, then refresh that same
    // safe setpoint until the first high-level target arrives.
    if (servoKeepalive && result == DG_RESULT_NONE)
    {
        result = MoveServoJoint(_tempPos);
        _lastMotionResult.store(static_cast<int>(result));
    }

    // --------------------------

	for(int i=0;i<MAX_JOINT_COUNT;i++)
	{
		_P[i] = 4.0f;   // типичные безопасные значения
		_D[i] = 2.5f;
	}

    SetJointGainPAll(_P);
	SetJointGainDAll(_D);

    SetMotionTimeAllEqual(300);

    // --------------------------

    // A DGControl instance is a process-wide singleton.  Never replay a
    // target left from a previous start/stop cycle.
    while (_target_joint_buffer.pop(_msgTargetPos)) {}

    _control = std::jthread(&DGControl::_loop, this);
}

void DGControl::stop()
{
    // Do not send a position command during shutdown.  In particular, a
    // DISARMED high-level controller must never cause an implicit movement.
    _g_connected.store(false);
    if (_control.joinable()) _control.join();
    _controlRunning.store(false);
    _temperatureSafe.store(false);
    _lastTelemetryNs.store(0);
    _g_commPeriod.store(0);
    if (_systemStarted.exchange(false)) SystemStop();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    DisconnectToGripper();

    std::cout << "Disconnected\n";
}

// --------------------------------------------------------------------------------

void DGControl::_loop()
{
    // DGSDK documents the callback value as communication frequency in Hz,
    // not as a duration in microseconds.  Cap this adapter loop at 200 Hz;
    // Python supplies new position setpoints at 50 Hz.
    constexpr int maxControlRateHz = 200;
    const int reportedRateHz = std::max(1, _g_commPeriod.load());
    const int controlRateHz = std::min(reportedRateHz, maxControlRateHz);
    const auto controlPeriod = std::chrono::nanoseconds(
        1'000'000'000LL / controlRateHz
    );

    std::cout << "DGControl loop: SDK=" << reportedRateHz
              << " Hz, adapter=" << controlRateHz << " Hz\n";

    auto next = std::chrono::steady_clock::now();
    _controlRunning.store(true);

    while(_g_connected.load())
    {
        {
            std::lock_guard<std::mutex> lock(_gripperDataMutex);
            std::memcpy(_currentPos, _g_gripperData.joint, sizeof(_g_gripperData.joint));
            std::memcpy(_currentCur, _g_gripperData.current, sizeof(_g_gripperData.current));
            std::memcpy(_currentVel, _g_gripperData.velocity, sizeof(_g_gripperData.velocity));
            std::memcpy(_currentTemp, _g_gripperData.temperature, sizeof(_g_gripperData.temperature));
        }

        array2EigenArray(_currentPos, _msgCurrentPos);
        array2EigenArray(_currentCur, _msgCurrentCur);
        array2EigenArray(_currentVel, _msgCurrentVel);
        array2EigenArray(_currentTemp, _msgCurrentTemp);

        _current_joint_buffer.push(_msgCurrentPos);
        _current_current_buffer.push(_msgCurrentCur);
        _current_velocity_buffer.push(_msgCurrentVel);
        _current_temperature_buffer.push(_msgCurrentTemp);

        const bool temperatureSafe = _checkTemp();
        _temperatureSafe.store(temperatureSafe);

        // Always consume the mailbox.  If motion is unsafe, discard pending
        // targets so they cannot execute later after the fault clears.
        bool hasTarget = false;
        while (_target_joint_buffer.pop(_msgTargetPos))
        {
            hasTarget = true;
        }

        // Do not call MoveServoJoint before the first accepted target.  This
        // is what makes connect() and DISARMED genuinely motion-free.
        if (hasTarget && temperatureSafe)
        {
            eigenArray2Array(_msgTargetPos, _targetPos);
            _updatePos();
        }

        if (temperatureSafe && (hasTarget || _servoKeepaliveEnabled.load()))
        {
            _tempPos[16] = 0.0;         // Зануление для безопасности дефектного 16 джоинта
            const DG_RESULT result = MoveServoJoint(_tempPos);
            _lastMotionResult.store(static_cast<int>(result));
            if (result == DG_RESULT_NONE)
            {
                std::lock_guard<std::mutex> lock(_commandMutex);
                std::memcpy(_latestCommandPos, _tempPos, sizeof(_tempPos));
                _latestCommandValid.store(true);
            }
        }

        next += controlPeriod;
        const auto now = std::chrono::steady_clock::now();
        if (next < now)
        {
            // Do not run catch-up iterations back-to-back after an overrun.
            next = now;
        }
        std::this_thread::sleep_until(next);
    }

    _controlRunning.store(false);
    _temperatureSafe.store(false);
}

void DGControl::_updatePos()
{
    for (int8_t i = 0; i < MAX_JOINT_COUNT; ++i)
    {
        // MoveServoJoint is the real-time SDK command.  Passing the latest
        // valid setpoint directly avoids adding a software ramp before the
        // gripper's own closed-loop response.
        if ((_targetPos[i] >= _lowerLimits[i] - _deltaLimits) &&
            (_targetPos[i] <= _upperLimits[i] + _deltaLimits))
        {
            _tempPos[i] = _targetPos[i];
        }
    }
}

bool DGControl::_checkTemp()
{
    for(int8_t i = 0; i < MAX_JOINT_COUNT; ++i)
    {
        if(_currentTemp[i] >= _tempLimit) return false;
    }
    return true;
}

// --------------------------------------------------------------------------------

bool DGControl::setTragetPosition(const Eigen::Array<double,MAX_JOINT_COUNT,1> &position)
{
    if (!_g_connected.load() || !_controlRunning.load() ||
        !_temperatureSafe.load() ||
        _lastMotionResult.load() != DG_RESULT_NONE)
    {
        return false;
    }
    return _target_joint_buffer.push(position);
}

bool DGControl::getCurrentPosition(Eigen::Array<double,MAX_JOINT_COUNT,1> &position)
{
    return _current_joint_buffer.pop(position);
}

bool DGControl::getCurrentCurrent(Eigen::Array<double,MAX_JOINT_COUNT,1> &current)
{
    return _current_current_buffer.pop(current);
}

bool DGControl::getCurrentVelocity(Eigen::Array<double,MAX_JOINT_COUNT,1> &velocity)
{
    return _current_velocity_buffer.pop(velocity);
}

bool DGControl::getCurrentTemperature(Eigen::Array<double,MAX_JOINT_COUNT,1> &temperature)
{
    return _current_temperature_buffer.pop(temperature);
}

// -----------------------

bool DGControl::setTragetPosition(const float* position)
{
    if (!_g_connected.load() || !_controlRunning.load() ||
        !_temperatureSafe.load() ||
        _lastMotionResult.load() != DG_RESULT_NONE)
    {
        return false;
    }

    Eigen::Array<double,MAX_JOINT_COUNT,1> array;

    for(int8_t i = 0; i < MAX_JOINT_COUNT; ++i)
    {
        array(i) = position[i];
    }
    
    return _target_joint_buffer.push(array);
}

bool DGControl::isConnected() const
{
    return _g_connected.load();
}

bool DGControl::isControlRunning() const
{
    return _controlRunning.load();
}

bool DGControl::isSystemStarted() const
{
    return _systemStarted.load();
}

bool DGControl::isTemperatureSafe() const
{
    return _temperatureSafe.load();
}

bool DGControl::isServoKeepaliveEnabled() const
{
    return _servoKeepaliveEnabled.load();
}

bool DGControl::isMotionReady() const
{
    return _g_connected.load() &&
           _controlRunning.load() &&
           _systemStarted.load() &&
           isTelemetryValid() &&
           _temperatureSafe.load() &&
           _lastMotionResult.load() == DG_RESULT_NONE;
}

bool DGControl::isTelemetryValid() const
{
    const std::int64_t last = _lastTelemetryNs.load();
    constexpr std::int64_t telemetryTimeoutNs = 500'000'000LL;
    return _g_connected.load() && last > 0 &&
           steadyNowNs() - last <= telemetryTimeoutNs;
}

int DGControl::getCommunicationRateHz() const
{
    return _g_commPeriod.load();
}

int DGControl::getDataProcessingStatus() const
{
    return _g_processing.load();
}

int DGControl::getLastMotionResult() const
{
    return _lastMotionResult.load();
}

std::uint64_t DGControl::getDisconnectCount() const
{
    return _disconnectCount.load();
}

std::uint64_t DGControl::getReconnectCount() const
{
    return _reconnectCount.load();
}

int DGControl::getDiagnosisProcess() const { return _diagnosisProcess.load(); }
int DGControl::getDiagnosisStep() const { return _diagnosisStep.load(); }
int DGControl::getDiagnosisJointId() const { return _diagnosisJointId.load(); }
int DGControl::getDiagnosisPeriod() const { return _diagnosisPeriod.load(); }
int DGControl::getDiagnosisJoint() const { return _diagnosisJoint.load(); }
int DGControl::getDiagnosisTemperature() const
{
    return _diagnosisTemperature.load();
}

bool DGControl::getLatestCommand(float* command) const
{
    if (!_latestCommandValid.load()) return false;
    std::lock_guard<std::mutex> lock(_commandMutex);
    std::memcpy(command, _latestCommandPos, sizeof(_latestCommandPos));
    return true;
}

bool DGControl::getCurrentPosition(float* position)
{
    Eigen::Array<double,MAX_JOINT_COUNT,1> array;
    if (_current_joint_buffer.pop(array))
    {
        for(int8_t i = 0; i < MAX_JOINT_COUNT; ++i)
        {
            position[i] = array[i];
        }
        return true;
    }
    else
    {
        return false;
    }
}

bool DGControl::getCurrentCurrent(float* current)
{
    Eigen::Array<double,MAX_JOINT_COUNT,1> array;
    if (_current_current_buffer.pop(array))
    {
        for(int8_t i = 0; i < MAX_JOINT_COUNT; ++i)
        {
            current[i] = array[i];
        }
        return true;
    }
    else
    {
        return false;
    }
}

bool DGControl::getCurrentVelocity(float* velocity)
{
    Eigen::Array<double,MAX_JOINT_COUNT,1> array;
    if (_current_velocity_buffer.pop(array))
    {
        for(int8_t i = 0; i < MAX_JOINT_COUNT; ++i)
        {
            velocity[i] = array[i];
        }
        return true;
    }
    else
    {
        return false;
    }
}

bool DGControl::getCurrentTemperature(float* temperature)
{
    Eigen::Array<double,MAX_JOINT_COUNT,1> array;
    if (_current_temperature_buffer.pop(array))
    {
        for(int8_t i = 0; i < MAX_JOINT_COUNT; ++i)
        {
            temperature[i] = array[i];
        }
        return true;
    }
    else
    {
        return false;
    }
}

// --------------------------------------------------------------------------------

void handcontrol::eigenArray2Array(const Eigen::Array<double,MAX_JOINT_COUNT,1> &eigen_array, float* array)
{
    for (int8_t i = 0; i < MAX_JOINT_COUNT; ++i)
    {
        array[i] = eigen_array[i];
    }
}

void handcontrol::array2EigenArray(float* array, Eigen::Array<double,MAX_JOINT_COUNT,1> &eigen_array)
{
    for (int8_t i = 0; i < MAX_JOINT_COUNT; ++i)
    {
        eigen_array[i] = array[i];
    }
}

int handcontrol::sign(float a, float threshold)
{
    return (a > std::abs(threshold)) ? 1 : ((a < -std::abs(threshold)) ? -1 : 0);
}

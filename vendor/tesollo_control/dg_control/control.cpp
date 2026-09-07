#include "control.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>

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
    std::snprintf(this->_ip, sizeof(this->_ip), "%s", ip);

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
        control->_recoveryRequired.store(true);
        control->_systemStarted.store(false);
        control->_systemStartNs.store(0);
        control->_lastTelemetryNs.store(0);
        control->_latestCommandValid.store(false);
        std::cerr << "DGSDK disconnected callback\n";
    }
}

void DGControl::_ReceivedGripperDataCallback(ReceivedGripperData data)
{
    DGControl* control = getInstance();
    {
        std::lock_guard<std::mutex> lock(control->_gripperDataMutex);
        control->_g_gripperData = data;
        // DGSDK can invoke this callback before SystemStart with a zero-filled
        // structure.  Keep the raw snapshot for diagnostics, but never call it
        // valid motor telemetry until SystemStart has succeeded.
        if (control->_systemStarted.load())
        {
            bool safe = true;
            for (int i = 0; i < MAX_JOINT_COUNT; ++i)
                safe = safe && std::isfinite(data.temperature[i]) &&
                       data.temperature[i] < control->_tempLimit;
            control->_temperatureSafe.store(safe);
            control->_lastTelemetryNs.store(steadyNowNs());
        }
    }
}

void DGControl::_CommunicationPeriodCallback(int period)
{
    getInstance()->_g_commPeriod = period;
    getInstance()->_lastCommunicationNs.store(steadyNowNs());
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
    if (_control.joinable()) throw std::runtime_error("DGControl already started");
    _stopRequested.store(false);
    _recoveryRequired.store(false);
    _temperatureSafe.store(false);
    _latestCommandValid.store(false);
    _lastTelemetryNs.store(0);
    _systemStartNs.store(0);
    _systemStarted.store(false);
    _lastMotionResult.store(DG_RESULT_NONE);
    _servoKeepaliveEnabled.store(servoKeepalive);

    GripperSystemSetting setting{};
    setting.communicationMode = COMMUNICATION_MODE_ETHERNET;
    setting.controlMode       = CONTROL_MODE_DEVELOPER;
    setting.port              = this->_port;
    setting.slaveID           = this->_slaveID;
    setting.readTimeout       = this->_readTimeout;
    std::memcpy(setting.ip, this->_ip, MAX_GRIPPER_IP_ADDRESS_SIZE);

    DG_RESULT result = SetGripperSystem(setting);
    std::cout << "SetGripperSystem: " << result << "\n";
    if (result != DG_RESULT_NONE)
        throw std::runtime_error("DGSDK SetGripperSystem failed: " + std::to_string(result));

    this->_setCallbacks();

    result = ConnectToGripper();
    std::cout << "ConnectToGripper: " << result << "\n";
    if (result != DG_RESULT_NONE)
        throw std::runtime_error("DGSDK ConnectToGripper failed: " + std::to_string(result));

    const auto connectDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!_g_connected.load())
    {
        if (std::chrono::steady_clock::now() >= connectDeadline)
            throw std::runtime_error("DGSDK connection timeout");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    GripperSetting gs{};
    gs.model = DG_MODEL_DG_5F_RIGHT;
    gs.movingInpose = 1;
    int type[MAX_RECEIVED_DATA_TYPE_COUNT] = {1,2,3,4,5,6};
    std::memcpy(gs.receivedDataType, type, sizeof(type));

    result = SetGripperOption(gs);
    std::cout << "SetGripperOption: " << result << "\n";
    if (result != DG_RESULT_NONE)
        throw std::runtime_error("DGSDK SetGripperOption failed: " + std::to_string(result));

    // This old DGSDK performs model/transport discovery asynchronously after
    // SetGripperOption.  The original working driver waited for a healthy
    // communication-rate callback before SystemStart; calling SystemStart too
    // early can return DG_RESULT_NOT_FOUND_MODEL (111).
    const auto readyDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (_g_commPeriod.load() < 200)
    {
        if (!_g_connected.load())
            throw std::runtime_error("DGSDK disconnected during model discovery");
        if (std::chrono::steady_clock::now() >= readyDeadline)
            throw std::runtime_error(
                "DGSDK model/communication discovery timeout (rate=" +
                std::to_string(_g_commPeriod.load()) + " Hz)"
            );
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    // Reject every pre-SystemStart zero/stale callback.  Only a callback that
    // arrives after successful SystemStart may seed the physical pose.
    _lastTelemetryNs.store(0);
    _temperatureSafe.store(false);
    _systemStartNs.store(steadyNowNs());
    result = SystemStart();
    std::cout << "SystemStart: " << result << "\n";
    if (result != DG_RESULT_NONE)
    {
        _systemStartNs.store(0);
        throw std::runtime_error("DGSDK SystemStart failed: " + std::to_string(result));
    }
    _systemStarted.store(true);

    const auto telemetryDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!isTelemetryValid())
    {
        if (!_g_connected.load())
            throw std::runtime_error("DGSDK disconnected before post-SystemStart telemetry");
        if (std::chrono::steady_clock::now() >= telemetryDeadline)
            throw std::runtime_error("DGSDK post-SystemStart telemetry timeout");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    // We now have a real physical sample.  Seed the future command state from
    // it, but DO NOT send MoveServoJoint while the bridge is DISARMED.
    {
        std::lock_guard<std::mutex> lock(_gripperDataMutex);
        for (int i = 0; i < MAX_JOINT_COUNT; ++i)
        {
            if (!std::isfinite(_g_gripperData.joint[i]))
                throw std::runtime_error("DGSDK returned invalid joint telemetry");
            _tempPos[i] = _g_gripperData.joint[i];
        }
    }
    _tempPos[16] = 0.0f;
    if (!_temperatureSafe.load())
        throw std::runtime_error("DGSDK temperature unsafe after SystemStart");

    for (int i = 0; i < MAX_JOINT_COUNT; ++i)
    {
        _P[i] = 4.0f;
        _D[i] = 2.5f;
    }
    SetJointGainPAll(_P);
    SetJointGainDAll(_D);
    SetMotionTimeAllEqual(300);

    while (_target_joint_buffer.pop(_msgTargetPos)) {}

    _control = std::jthread(&DGControl::_loop, this);
}

void DGControl::stop()
{
    // Do not send a position command during shutdown.  In particular, a
    // DISARMED high-level controller must never cause an implicit movement.
    _stopRequested.store(true);
    if (_control.joinable()) _control.join();
    _controlRunning.store(false);
    _latestCommandValid.store(false);
    _lastTelemetryNs.store(0);
    _systemStartNs.store(0);
    _g_commPeriod.store(0);
    if (_systemStarted.exchange(false)) SystemStop();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    DisconnectToGripper();
    _g_connected.store(false);

    std::cout << "Disconnected\n";
}

// --------------------------------------------------------------------------------

void DGControl::_loop()
{
    // The DGSDK communication-rate callback is asynchronous and is often still
    // zero/1 Hz when this thread starts.  Sampling it once here previously
    // latched the servo/keepalive loop at 1 Hz for the lifetime of the process.
    // That is especially dangerous with the latest-target mailbox: Python may
    // enqueue many guarded 5-degree steps at 50 Hz, while a 1 Hz consumer drains
    // them all and sends only the newest target as one large physical jump.
    //
    // Run the adapter independently at a stable 200 Hz.  DGSDK telemetry may be
    // ~400-800 Hz and Python targets are ~50 Hz; repeating the latest safe servo
    // setpoint at 200 Hz is the intended keepalive behaviour.
    constexpr int controlRateHz = 200;
    constexpr auto controlPeriod = std::chrono::nanoseconds(5'000'000LL);

    std::cout << "DGControl loop: adapter=" << controlRateHz
              << " Hz (SDK communication rate is asynchronous)\n";

    auto next = std::chrono::steady_clock::now();
    _controlRunning.store(true);

    while(!_stopRequested.load())
    {
        std::unique_lock<std::mutex> motionLock(_motionMutex);
        if (!_g_connected.load())
        {
            while (_target_joint_buffer.pop(_msgTargetPos)) {}
            motionLock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            next = std::chrono::steady_clock::now();
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(_gripperDataMutex);
            for (int i = 0; i < MAX_JOINT_COUNT; ++i)
            {
                _currentPos[i] = static_cast<float>(_g_gripperData.joint[i]);
                _currentCur[i] = static_cast<float>(_g_gripperData.current[i]); // mA
                _currentVel[i] = static_cast<float>(_g_gripperData.velocity[i]); // rpm
                _currentTemp[i] = static_cast<float>(_g_gripperData.temperature[i]); // C
            }
        }

        array2EigenArray(_currentPos, _msgCurrentPos);
        array2EigenArray(_currentCur, _msgCurrentCur);
        array2EigenArray(_currentVel, _msgCurrentVel);
        array2EigenArray(_currentTemp, _msgCurrentTemp);

        _current_joint_buffer.push(_msgCurrentPos);
        _current_current_buffer.push(_msgCurrentCur);
        _current_velocity_buffer.push(_msgCurrentVel);
        _current_temperature_buffer.push(_msgCurrentTemp);

        const bool temperatureSafe = _temperatureSafe.load();
        if (!isTelemetryValid() || !temperatureSafe || _lastMotionResult.load() != DG_RESULT_NONE)
            _recoveryRequired.store(true);

        // Always consume the mailbox.  If motion is unsafe, discard pending
        // targets so they cannot execute later after the fault clears.
        bool hasTarget = false;
        while (_target_joint_buffer.pop(_msgTargetPos))
        {
            hasTarget = true;
        }

        // Do not call MoveServoJoint before the first accepted target.  This
        // is what makes connect() and DISARMED genuinely motion-free.
        const bool mayMove = isMotionReady();
        if (hasTarget && mayMove)
        {
            eigenArray2Array(_msgTargetPos, _targetPos);
            _updatePos();
        }

        // DISARMED is physically silent.  Keepalive only becomes active
        // after at least one high-level target has been accepted and sent.
        const bool keepaliveActive =
            _servoKeepaliveEnabled.load() && _latestCommandValid.load();
        if (mayMove && (hasTarget || keepaliveActive))
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

        motionLock.unlock();

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
    std::lock_guard<std::mutex> lock(_motionMutex);
    if (!isMotionReady() || !position.isFinite().all())
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
    Eigen::Array<double,MAX_JOINT_COUNT,1> array;

    for(int8_t i = 0; i < MAX_JOINT_COUNT; ++i)
    {
        array(i) = position[i];
    }
    
    return setTragetPosition(array);
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
    return motionReadyReason() == "READY";
}

std::string DGControl::motionReadyReason() const
{
    if (!_g_connected.load()) return "DISCONNECTED";
    if (!_controlRunning.load()) return "CONTROL_THREAD_STOPPED";
    if (!isTelemetryValid()) return "TELEMETRY_STALE";
    if (!_temperatureSafe.load()) return "TEMPERATURE_UNSAFE";
    if (!_systemStarted.load()) return "SYSTEM_NOT_STARTED";
    if (_lastMotionResult.load() != DG_RESULT_NONE) return "MOTION_RESULT_ERROR";
    if (_recoveryRequired.load()) return "RECOVERY_REQUIRED";
    return "READY";
}

double DGControl::telemetryAgeMs() const
{
    const auto last = _lastTelemetryNs.load();
    return last > 0 ? (steadyNowNs() - last) / 1e6 : std::numeric_limits<double>::quiet_NaN();
}

double DGControl::communicationAgeMs() const
{
    const auto last = _lastCommunicationNs.load();
    return last > 0 ? (steadyNowNs() - last) / 1e6 : std::numeric_limits<double>::quiet_NaN();
}

bool DGControl::getTelemetry(float* position, float* current, float* velocity, float* temperature) const
{
    std::lock_guard<std::mutex> lock(_gripperDataMutex);
    if (_lastTelemetryNs.load() == 0) return false;
    for (int i = 0; i < MAX_JOINT_COUNT; ++i)
    {
        position[i] = static_cast<float>(_g_gripperData.joint[i]);
        current[i] = static_cast<float>(_g_gripperData.current[i]);
        velocity[i] = static_cast<float>(_g_gripperData.velocity[i]);
        temperature[i] = static_cast<float>(_g_gripperData.temperature[i]);
    }
    return true;
}

void DGControl::recover(float* measuredPose, double timeoutSeconds)
{
    // Explicit recovery only; ARM never calls this path.  Recovery itself is
    // motion-free and leaves the bridge DISARMED.
    std::lock_guard<std::mutex> lock(_motionMutex);
    _recoveryRequired.store(true);
    _latestCommandValid.store(false);
    while (_target_joint_buffer.pop(_msgTargetPos)) {}

    if (!_controlRunning.load())
        throw std::runtime_error("control thread stopped");
    if (!std::isfinite(timeoutSeconds) || timeoutSeconds <= 0.0 || timeoutSeconds > 10.0)
        throw std::runtime_error("invalid recovery timeout");

    const auto deadlineAfter = [timeoutSeconds]() {
        return std::chrono::steady_clock::now() +
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(timeoutSeconds));
    };

    // The vendor library normally reconnects its transport asynchronously.
    // Do not race it with DisconnectToGripper/ConnectToGripper here.  Wait for
    // the existing transport to become usable; if it does not, fail safely and
    // let the user restart the hardware pipeline.
    {
        const auto deadline = deadlineAfter();
        while (!_g_connected.load())
        {
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error(
                    "DGSDK transport did not reconnect; restart hardware pipeline"
                );
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    if (_systemStarted.exchange(false))
    {
        const DG_RESULT stopResult = SystemStop();
        if (stopResult != DG_RESULT_NONE)
            throw std::runtime_error(
                "SystemStop failed during recovery: " + std::to_string(stopResult)
            );
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Require a fresh communication/model-discovery signal for this recovery
    // generation rather than reusing the rate cached before the fault.
    _g_commPeriod.store(0);
    _lastCommunicationNs.store(0);

    GripperSetting gs{};
    gs.model = DG_MODEL_DG_5F_RIGHT;
    gs.movingInpose = 1;
    int type[MAX_RECEIVED_DATA_TYPE_COUNT] = {1,2,3,4,5,6};
    std::memcpy(gs.receivedDataType, type, sizeof(type));
    const DG_RESULT optionResult = SetGripperOption(gs);
    if (optionResult != DG_RESULT_NONE)
        throw std::runtime_error(
            "SetGripperOption failed during recovery: " + std::to_string(optionResult)
        );

    // Wait for the same asynchronous model-discovery gate that the known-good
    // startup path requires before SystemStart.
    {
        const auto deadline = deadlineAfter();
        while (_g_commPeriod.load() < 200)
        {
            if (!_g_connected.load())
                throw std::runtime_error("DGSDK disconnected during recovery discovery");
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error(
                    "DGSDK recovery model/communication discovery timeout (rate=" +
                    std::to_string(_g_commPeriod.load()) + " Hz)"
                );
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    _lastTelemetryNs.store(0);
    _temperatureSafe.store(false);
    _systemStartNs.store(steadyNowNs());
    _lastMotionResult.store(DG_RESULT_NONE);

    const DG_RESULT startResult = SystemStart();
    if (startResult != DG_RESULT_NONE)
    {
        _systemStartNs.store(0);
        throw std::runtime_error(
            "SystemStart failed during recovery: " + std::to_string(startResult)
        );
    }
    _systemStarted.store(true);

    // Wait only for telemetry generated by this SystemStart generation.
    {
        const auto deadline = deadlineAfter();
        while (!isTelemetryValid())
        {
            if (!_g_connected.load())
                throw std::runtime_error("DGSDK disconnected while recovering telemetry");
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("DGSDK post-recovery telemetry timeout");
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    {
        std::lock_guard<std::mutex> dataLock(_gripperDataMutex);
        for (int i = 0; i < MAX_JOINT_COUNT; ++i)
        {
            if (!std::isfinite(_g_gripperData.joint[i]))
                throw std::runtime_error("invalid measured pose after recovery");
            if (!std::isfinite(_g_gripperData.temperature[i]) ||
                _g_gripperData.temperature[i] >= _tempLimit)
                throw std::runtime_error("temperature unsafe after recovery");
            _tempPos[i] = _g_gripperData.joint[i];
            measuredPose[i] = _g_gripperData.joint[i];
        }
    }
    _tempPos[16] = 0.0f;
    measuredPose[16] = 0.0f;

    // No MoveServoJoint here.  The next explicit ARM starts a software blend
    // from this measured pose, and only then may the first servo command flow.
    _latestCommandValid.store(false);
    _recoveryRequired.store(false);
    if (!isMotionReady())
    {
        _recoveryRequired.store(true);
        throw std::runtime_error("hardware not ready after explicit recovery");
    }
}

void DGControl::suspendMotion()
{
    std::lock_guard<std::mutex> lock(_motionMutex);
    while (_target_joint_buffer.pop(_msgTargetPos)) {}
    _latestCommandValid.store(false);
}

bool DGControl::isTelemetryValid() const
{
    const std::int64_t last = _lastTelemetryNs.load();
    const std::int64_t started = _systemStartNs.load();
    constexpr std::int64_t telemetryTimeoutNs = 500'000'000LL;
    return _g_connected.load() && _systemStarted.load() &&
           started > 0 && last >= started &&
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

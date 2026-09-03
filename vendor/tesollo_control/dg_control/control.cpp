#include "control.hpp"

using namespace handcontrol;

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
    getInstance()->_g_connected.store(true);
}

void DGControl::_DisconnectedCallback()
{
    getInstance()->_g_connected.store(false);
}

void DGControl::_ReceivedGripperDataCallback(ReceivedGripperData data)
{
    getInstance()->_g_gripperData = data;
}

void DGControl::_CommunicationPeriodCallback(int period)
{
    getInstance()->_g_commPeriod = period;
}

void DGControl::_DiagnosisCallback(DiagnosisSystem diag)
{
    getInstance()->_g_diagnosisData = diag;
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

void DGControl::start()
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

    result = SystemStart();
    std::cout << "SystemStart: " << result << "\n";
    success += result;

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

    std::memcpy(_tempPos, _g_gripperData.joint, sizeof(_g_gripperData.joint));

    _control = std::jthread(&DGControl::_loop, this);
}

void DGControl::stop()
{
    if (_g_connected.load()) 
    {
        MoveJointAll(_currentPos); // Чтобы пальцы после выключения не двигались

        _g_connected.store(false);
        if (_control.joinable()) _control.join();
        _g_commPeriod.store(0);
        SystemStop();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        DisconnectToGripper();

        std::cout << "Disconnected\n"; 
    }
}

// --------------------------------------------------------------------------------

void DGControl::_loop()
{
    auto next = std::chrono::steady_clock::now();

    while(_g_connected.load())
    {
        std::memcpy(_currentPos, _g_gripperData.joint, sizeof(_g_gripperData.joint));   // Запись информации с гриппера
        std::memcpy(_currentCur, _g_gripperData.current, sizeof(_g_gripperData.current));   // Запись информации с гриппера
        std::memcpy(_currentVel, _g_gripperData.velocity, sizeof(_g_gripperData.velocity));   // Запись информации с гриппера
        std::memcpy(_currentTemp, _g_gripperData.temperature, sizeof(_g_gripperData.temperature));   // Запись информации с гриппера

        array2EigenArray(_currentPos, _msgCurrentPos);
        array2EigenArray(_currentCur, _msgCurrentCur);
        array2EigenArray(_currentVel, _msgCurrentVel);
        array2EigenArray(_currentTemp, _msgCurrentTemp);

        _current_joint_buffer.push(_msgCurrentPos);
        _current_current_buffer.push(_msgCurrentCur);
        _current_velocity_buffer.push(_msgCurrentVel);
        _current_temperature_buffer.push(_msgCurrentTemp);

        if (_checkTemp())
        {
            // Position commands are setpoints, not a trajectory queue.  Drain
            // all pending commands so a temporary producer/consumer mismatch
            // cannot build latency by replaying stale targets.
            bool hasTarget = false;
            while (_target_joint_buffer.pop(_msgTargetPos))
            {
                hasTarget = true;
            }
            if (hasTarget) eigenArray2Array(_msgTargetPos, _targetPos);

            _updatePos();

            _tempPos[16] = 0.0;         // Зануление для безопасности дефектного 16 джоинта
            MoveServoJoint(_tempPos); 
        }
        

        next += std::chrono::microseconds(_g_commPeriod);
        const auto now = std::chrono::steady_clock::now();
        if (next < now)
        {
            // Do not run catch-up iterations back-to-back after an overrun.
            next = now;
        }
        std::this_thread::sleep_until(next);
    }
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
    
    return _target_joint_buffer.push(array);
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

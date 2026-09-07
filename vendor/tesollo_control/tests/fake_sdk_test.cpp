// Links against control.cpp and fake functions ONLY, never libDGSDK or a socket.
#include "../dg_control/control.hpp"
#include <cassert>
#include <vector>
#include <cmath>

using namespace handcontrol;
using namespace std::chrono_literals;
static ConnectedToGripperCallback connected;
static DisconnectedToGripperCallback disconnected;
static ReceivedGripperDatasCallback received;
static CommunicationPeriodCallback rate;
static std::atomic<bool> packets{true};
static std::atomic<int> moves{0};
static std::atomic<int> systemStarts{0};
static std::atomic<float> samplePosition{10.0f};
static std::mutex sentMutex;
static float sent[20];
static ReceivedGripperData sample(float position=10.0f) {
    ReceivedGripperData data{};
    for(int i=0;i<20;++i) {
        data.joint[i]=position; data.current[i]=311; data.velocity[i]=-12;
        data.temperature[i]=42.5f;
    }
    return data;
}
DG_RESULT SetGripperSystem(GripperSystemSetting) { return DG_RESULT_NONE; }
DG_RESULT SetGripperOption(GripperSetting) {
    if (rate) rate(430);
    return DG_RESULT_NONE;
}
DG_RESULT ConnectToGripper() {
    connected();
    // Model/transport discovery happens before SystemStart in the real SDK.
    rate(430);
    // A pre-SystemStart zero packet must never be treated as physical data.
    if (packets.load()) received(ReceivedGripperData{});
    return DG_RESULT_NONE;
}
DG_RESULT DisconnectToGripper() { disconnected(); return DG_RESULT_NONE; }
DG_RESULT SystemStart() {
    ++systemStarts;
    if (packets.load()) {
        std::thread([] {
            std::this_thread::sleep_for(2ms);
            if (packets.load()) received(sample(samplePosition.load()));
        }).detach();
    }
    return DG_RESULT_NONE;
}
DG_RESULT SystemStop() { return DG_RESULT_NONE; }
DG_RESULT SetJointGainPAll(float*) { return DG_RESULT_NONE; }
DG_RESULT SetJointGainDAll(float*) { return DG_RESULT_NONE; }
DG_RESULT SetMotionTimeAllEqual(int) { return DG_RESULT_NONE; }
DG_RESULT MoveServoJoint(float* command) {
    std::lock_guard<std::mutex> lock(sentMutex);
    std::copy(command,command+20,sent); ++moves; return DG_RESULT_NONE;
}
DG_RESULT CallbackForOnConnected(ConnectedToGripperCallback cb) { connected=cb; return DG_RESULT_NONE; }
DG_RESULT CallbackForOnDisconnected(DisconnectedToGripperCallback cb) { disconnected=cb; return DG_RESULT_NONE; }
DG_RESULT CallbackForOnReceivedGripperData(ReceivedGripperDatasCallback cb) { received=cb; return DG_RESULT_NONE; }
DG_RESULT CallbackForOnCommunicationPeriod(CommunicationPeriodCallback cb) { rate=cb; return DG_RESULT_NONE; }
DG_RESULT CallbackForOnDiagnosisSystem(DiagnosisSystemCallback) { return DG_RESULT_NONE; }
DG_RESULT CallbackForOnReceivedFingertipSensorData(ReceivedSensorCallback) { return DG_RESULT_NONE; }
DG_RESULT CallbackForOnReceivedGPIOData(ReceivedGPIOCallback) { return DG_RESULT_NONE; }
DG_RESULT CallbackForOnDataProcessing(DataProcessingCallback) { return DG_RESULT_NONE; }

int main() {
    auto* control=DGControl::getInstance("127.0.0.1",502,1);
    control->start();
    // Startup is motion-free: pre-SystemStart zero telemetry is ignored and
    // no MoveServoJoint is emitted while the bridge is DISARMED.
    assert(moves.load() == 0);
    std::this_thread::sleep_for(30ms);
    assert(moves.load() == 0);
    float p[20], c[20], v[20], t[20];
    assert(control->getTelemetry(p,c,v,t));
    assert(c[0]==311.0f && v[0]==-12.0f && t[0]==42.5f && p[0]==10.0f);
    assert(control->getCurrentCurrent(c) && c[0]==311.0f);
    assert(control->getCurrentVelocity(v) && v[0]==-12.0f);
    assert(control->isTemperatureSafe());
    float oldTarget[20]; std::fill(oldTarget,oldTarget+20,30.0f);
    assert(control->setTragetPosition(oldTarget));
    std::this_thread::sleep_for(30ms);
    assert(moves.load() >= 1);
    disconnected(); disconnected(); // duplicate callback must not increment again
    std::this_thread::sleep_for(30ms);
    assert(control->getDisconnectCount()==1);
    assert(control->isControlRunning());
    assert(!control->isMotionReady());
    assert(control->motionReadyReason()=="DISCONNECTED");
    assert(control->isTemperatureSafe());
    auto before=moves.load();
    samplePosition.store(15.0f);
    connected(); rate(430);
    std::this_thread::sleep_for(30ms);
    assert(moves==before); // reconnect alone cannot replay stale output
    assert(!control->setTragetPosition(oldTarget));
    assert(control->getReconnectCount()==1);
    control->recover(p,0.5);
    assert(p[0]==15.0f && p[16]==0.0f && control->isMotionReady());
    assert(moves==before); // recovery itself is motion-free
    packets=false;
    std::this_thread::sleep_for(550ms);
    assert(!control->isTelemetryValid() && control->isTemperatureSafe());
    assert(control->motionReadyReason()=="TELEMETRY_STALE");

    // Simulate a reconnect whose restarted system never produces telemetry.
    // Recovery may call SystemStart, but must never emit a servo command.
    disconnected();
    connected(); rate(430);
    before=moves;
    const auto startsBeforeFailedRecovery = systemStarts.load();
    bool failed=false;
    try { control->recover(p,0.05); } catch(const std::runtime_error&) { failed=true; }
    assert(failed && moves==before);
    assert(systemStarts.load()==startsBeforeFailedRecovery + 1);
    auto hot=sample(); hot.temperature[2]=65.0f; received(hot);
    assert(!control->isTemperatureSafe());
    control->stop();
    assert(!control->isControlRunning());
    std::cout << "PASS: native telemetry conversion, persistent thread, stale-target gate, explicit recovery, thermal semantics\n";
}

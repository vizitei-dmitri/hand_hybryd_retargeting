#include <iostream>
#include <csignal>
#include "dg_control/control.hpp"
#include <chrono>
#include "udp/udp_server.hpp"
#include "poses/poses.hpp"

using namespace handcontrol;
using namespace server;

bool mainprog = true;

void signalHandler(int)
{
    mainprog = false;
}

int main()
{   
    DGControl* dg = DGControl::getInstance();

    // UDPServer<20,20> udp_server("192.168.68.201", 8081, "192.168.68.169", 8082);
    std::signal(SIGINT, signalHandler);

    Eigen::Array<double,MAX_JOINT_COUNT,1> pos;
    Eigen::Array<double,MAX_JOINT_COUNT,1> cur;
    Eigen::Array<double,MAX_JOINT_COUNT,1> vel;
    Eigen::Array<double,MAX_JOINT_COUNT,1> temp;

    Eigen::Array<double,MAX_JOINT_COUNT,1> target_pos1;
    Eigen::Array<double,MAX_JOINT_COUNT,1> target_pos2;

    pos << 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
    cur << 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
    vel << 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
    temp << 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;

    target_pos1 << 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
    target_pos2 << 0,0,50,50,
                0,0,50,50,
                0,0,50,50,
                0,0,50,50,
                0,0,50,50;
    
    dg->start();
    // udp_server.start();

    int pose = -1;

    std::chrono::steady_clock::time_point time = std::chrono::steady_clock::now();

    while(mainprog)
    {   
        if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - time).count() >= 1000)
        {
            ++pose;
            switch (pose)
            {
            case 0:
                dg->setTragetPosition(poses::numbers::FIVE);
                break;
            case 1:
                dg->setTragetPosition(poses::numbers::FOUR);
                break;
            case 2:
                dg->setTragetPosition(poses::numbers::THREE);
                break;
            case 3:
                dg->setTragetPosition(poses::numbers::TWO);
                break;
            case 4:
                dg->setTragetPosition(poses::numbers::ONE);
                break;
            case 5:
                dg->setTragetPosition(poses::numbers::ZERO);
                break;
            case 6:
                dg->setTragetPosition(poses::gestures::PEACE);
                break;
            case 7:
                dg->setTragetPosition(poses::gestures::FIXERS);
                break;
            case 8:
                dg->setTragetPosition(poses::gestures::GOAT);
                break;
            case 9:
                dg->setTragetPosition(poses::gestures::OKEY);
                break;
            case 10:
                dg->setTragetPosition(poses::grasps::OPEN);
                break;
            case 11:
                dg->setTragetPosition(poses::grasps::FIVE_FINGER_TO_POINT);
                break;
            case 12:
                dg->setTragetPosition(poses::grasps::THREE_FINGER_TO_POINT);
                break;
            case 13:
                dg->setTragetPosition(poses::grasps::THUMB_TO_INDEX);
                break;
            case 14:
                dg->setTragetPosition(poses::grasps::THUMB_TO_MIDDLE);
                break;
            default:
                dg->setTragetPosition(poses::numbers::FIVE);
                pose = 0;
                break;
            }
            time = std::chrono::steady_clock::now();
        }

        // if(udp_server.getMsg(target_pos1))
        // {
        //     dg->setTragetPosition(target_pos1);
        // }

        dg->getCurrentPosition(pos);
        dg->getCurrentCurrent(cur);
        dg->getCurrentVelocity(vel);
        dg->getCurrentTemperature(temp);

        // std::cout << pos[2] << " " << pos[3] << "\n";

        std::this_thread::sleep_for(std::chrono::microseconds(100));

    }

    dg->stop();
    // udp_server.stop();

    return 0;
}
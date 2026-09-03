#include "../dg_control/control.hpp"

namespace poses
{
    namespace grasps
    {
        constexpr float OPEN[MAX_JOINT_COUNT] = {0, -90, 40, 20,
                                            -20, 0, 60, 40,
                                            -10, 0, 60, 40,
                                            0, 0, 60, 40,
                                            10, 20, 45, 50};

        constexpr float OPEN_BIG[MAX_JOINT_COUNT] = {0, -90, 20, 10,
                                                -20, 0, 30, 20,
                                                -10, 0, 30, 20,
                                                0, 0, 30, 20,
                                                10, 20, 20, 25};

        constexpr float OPEN_VERY_BIG[MAX_JOINT_COUNT] = {0, -90, -25, 40,
                                                    -20, 0, 30, 20,
                                                    -10, 0, 30, 20,
                                                    0, 0, 30, 20,
                                                    10, 20, 0, 0};

        constexpr float FIVE_FINGER_TO_POINT[MAX_JOINT_COUNT] = {5, -95, 45, 30,
                                                            -15, 50, 50, 50,
                                                            -5, 30, 80, 30,
                                                            -10, 25, 70, 35,
                                                            35, 30, 70, 45};

        constexpr float THREE_FINGER_TO_POINT[MAX_JOINT_COUNT] = {-5, -90, 60, 20,
                                                            -15, 20, 75, 45,
                                                            -20, 15, 85, 35,
                                                            0, 0, 60, 40,
                                                            10, 20, 45, 50};

        constexpr float THUMB_TO_INDEX[MAX_JOINT_COUNT] = {-10, -90, 55, 25,
                                                        -20, 25, 75, 40,
                                                        -10, 0, 60, 40,
                                                        0, 0, 60, 40,
                                                        10, 20, 45, 50};

        constexpr float THUMB_TO_MIDDLE[MAX_JOINT_COUNT] = {5, -95, 55, 25,
                                                        -20, 0, 60, 40,
                                                        -12, 25, 85, 40,
                                                        0, 0, 60, 40,
                                                        10, 20, 45, 50};
        
        constexpr float FOUR_FINGER_TO_RECT[MAX_JOINT_COUNT] = {0, -90, -5, 50,
                                                            -10, 20, 50, 50,
                                                            0, 20, 50, 50,
                                                            0, 20, 50, 50,
                                                            10, 20, 0, 0};
    }

    namespace gestures
    {
        constexpr float START[MAX_JOINT_COUNT] = {0, 0, 0, 0,
                                                0, 0, 0, 0,
                                                0, 0, 0, 0,
                                                0, 0, 0, 0,
                                                0, 0, 0, 0};

        constexpr float GOAT[MAX_JOINT_COUNT] = {5, -5, 5, 10,
                                                -20, 0, 0, 0,
                                                -5, 70, 85, 85,
                                                3, 70, 85, 85,
                                                0, 20, 0, 0};

        constexpr float PEACE[MAX_JOINT_COUNT] = {0, -15, 60, 85,
                                                -20, 0, 0, 0,
                                                10, 0, 0, 0,
                                                15, 70, 85, 85,
                                                0, 20, 85, 85};

        constexpr float FIXERS[MAX_JOINT_COUNT] = {5, -5, 5, 10,
                                                -20, 0, 0, 0,
                                                10, 0, 0, 0,
                                                15, 70, 85, 85,
                                                0, 20, 85, 85};

        constexpr float MIDDLE_FINGER[MAX_JOINT_COUNT] = {-5, -35, 60, 25,
                                                        -20, 70, 85, 85,
                                                        0, 0, 0, 0,
                                                        15, 70, 85, 85,
                                                        0, 20, 85, 85};

        constexpr float OKEY[MAX_JOINT_COUNT] = {-5, -85, 40, 40,
                                                -5, 35, 60, 50,
                                                0, 0, 0, 0,
                                                5, 0, 0, 0,
                                                0, 15, 0, 0};

        constexpr float FIST[MAX_JOINT_COUNT] = {0, 0, 40, 55,
                                                0, 90, 80, 50,
                                                0, 90, 80, 25,
                                                0, 90, 80, 0,
                                                0, 0, 90, 90};    
                                                
        constexpr float INDEX[MAX_JOINT_COUNT] = {0, 0, 40, 55,
                                                0, 65, 0, 0,
                                                0, 90, 80, 25,
                                                0, 90, 80, 0,
                                                0, 0, 90, 90};

        constexpr float HOOK[MAX_JOINT_COUNT] = {0, 0, 40, 55,
                                                0, 25, 55, 40,
                                                0, 25, 55, 40,
                                                0, 25, 55, 40,
                                                0, 0, 30, 60};

        constexpr float HOOK_INDEX_1[MAX_JOINT_COUNT] = {0, 0, 40, 55,
                                                0, 0, 45, 85,
                                                0, 90, 80, 25,
                                                0, 90, 80, 0,
                                                0, 0, 90, 90};

        constexpr float HOOK_INDEX_2[MAX_JOINT_COUNT] = {0, 0, 40, 55,
                                                0, 45, 45, 85,
                                                0, 90, 80, 25,
                                                0, 90, 80, 0,
                                                0, 0, 90, 90};

        constexpr float OPEN_HOLDER_1[MAX_JOINT_COUNT] = {0, -90, 10, 40,
                                                0, 15, 55, 30,
                                                0, 0, 0, 0,
                                                5, 0, 0, 0,
                                                0, 15, 0, 0};

        constexpr float OPEN_HOLDER_2[MAX_JOINT_COUNT] = {5, -90, 10, 40,
                                                0, 55, 90, 45,
                                                0, 0, 0, 0,
                                                5, 0, 0, 0,
                                                0, 15, 0, 0};
    }

    namespace numbers
    {
        constexpr float ZERO[MAX_JOINT_COUNT] = {-5, -5, 60, 60,
                                                -10, 75, 85, 85,
                                                0, 75, 85, 85,
                                                5, 65, 85, 85,
                                                0, 15, 85, 85};

        constexpr float ONE[MAX_JOINT_COUNT] = {-5, -5, 60, 60,
                                                -10, 0, 0, 0,
                                                0, 75, 85, 85,
                                                5, 65, 85, 85,
                                                0, 15, 85, 85};

        constexpr float TWO[MAX_JOINT_COUNT] = {-5, -5, 60, 60,
                                                -10, 0, 0, 0,
                                                0, 0, 0, 0,
                                                5, 65, 85, 85,
                                                0, 15, 85, 85};

        constexpr float THREE[MAX_JOINT_COUNT] = {-5, -5, 60, 60,
                                                -10, 0, 0, 0,
                                                0, 0, 0, 0,
                                                5, 0, 0, 0,
                                                0, 15, 85, 85};

        constexpr float FOUR[MAX_JOINT_COUNT] = {-5, -5, 60, 60,
                                                -10, 0, 0, 0,
                                                0, 0, 0, 0,
                                                5, 0, 0, 0,
                                                0, 15, 0, 0};

        constexpr float FIVE[MAX_JOINT_COUNT] = {-5, -5, 25, 25,
                                                -10, 0, 0, 0,
                                                0, 0, 0, 0,
                                                5, 0, 0, 0,
                                                0, 15, 0, 0};
    }

    // namespace manipulations
    // {
    //     constexpr float PUSH[MAX_JOINT_COUNT] = {-5, -5, 60, 60,
    //                                             -10, 75, 85, 85,
    //                                             0, 75, 85, 85,
    //                                             5, 65, 85, 85,
    //                                             0, 15, 85, 85};
    // }
}
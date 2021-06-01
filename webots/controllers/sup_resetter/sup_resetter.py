"""my_controller1 controller."""

# https://cyberbotics.com/doc/guide/running-extern-robot-controllers?tab-language=python&tab-os=windows

# import numpy as np
# print(np.array([1,2]))
import json

from os import times
from controller import (
    Field,
    LightSensor,
    Node,
    PositionSensor,
    Robot,
    Motor,
    Keyboard,
    Camera,
    Supervisor,
    Emitter,
    Receiver,
)

TIMESTEP = 256

kb = Keyboard()

sup = Supervisor()

emit: Emitter = sup.getDevice("emitter")
rec: Receiver = sup.getDevice("receiver")
rec.enable(TIMESTEP)
robots = []



# iterate root and append all non-supervisor robots to robots list
root = sup.getRoot().getField("children")
for i in range(root.getCount()):
    n = root.getMFNode(i)
    if n.getBaseTypeName() == "Robot" and n.getField("supervisor").getSFBool() == False:
        robots.append(
            {
                "robot": n,
                "pos_0": n.getPosition(),
                "rot_0": n.getField("rotation").getSFRotation(),
                "last_pos": n.getPosition(),
                "idle_time": 0,
            }
        )
    if root.getMFNode(i) == root.getMFNode(-1):
        break


def reset_robot(robo, msg=""):
    r["robot"].getField("translation").setSFVec3f(r["pos_0"])
    r["robot"].getField("rotation").setSFRotation(r["rot_0"])
    r["robot"].resetPhysics()
    r["idle_time"] = 0
    bmsg = f'{{"reset":"{msg}"}}'
    emit.send(bmsg.encode())
    print(f"resetting robot ({msg}): {r['robot'].getDef()}")


while sup.step(TIMESTEP) != -1:
    for r in robots:
        r_trans: Field = r["robot"].getField("translation")
        values = r_trans.getSFVec3f()
        # if a robot falls below y-thresh, reset it
        if values[1] < -10:
            reset_robot(r, "falling")

        # reset robot if it is stuck too long
        travel_dis = (
            abs(values[0] - r["last_pos"][0])
            + abs(values[1] - r["last_pos"][1])
            + abs(values[1] - r["last_pos"][1])
        ) / 3
        if travel_dis < 0.01:
            r["idle_time"] += TIMESTEP
        else:
            r["idle_time"] = 0
            r["last_pos"] = r["robot"].getPosition()
        if r["idle_time"] > 10 * 1000:
            reset_robot(r, "idle")

        # listen for requests from robots
        while rec.getQueueLength() > 0:
            msg_dat = rec.getData()
            rec.nextPacket()


print("closing")
# Enter here exit cleanup code.

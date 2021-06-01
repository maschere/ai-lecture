"""spawner controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Supervisor,Emitter
from collections import deque
from random import random, choices
import numpy as np
import msgpack
# create the Robot instance.
sup = Supervisor()

emit: Emitter = sup.getDevice("emitter")

# get the time step of the current world.
timestep = 256
colors = [[1,0,0],[0,1,0],[0,0,1]]
names = ["red", "green", "blue"]
points = [1,-5,3]
# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getMotor('motorname')
#  ds = robot.getDistanceSensor('dsname')
#  ds.enable(timestep)
root = sup.getRoot().getField("children")

for i in range(root.getCount()):
    n = root.getMFNode(i)
    if n.getDef()=="epuckmod":
        robo = n

class Ball():
    def __init__(self, sec_alive = 20):
        self.idx = root.getCount()
        root.importMFNode(-1,"../../protos/Ball.wbo")
        self.node = root.getMFNode(self.idx)
        self.node.getField("translation").setSFVec3f([(random()-0.5),0.15,(random()-0.5)])
        self.ball_type = choices([0,1,2],[0.7,0.2,0.1])[0]
        self.node.getField("color").setSFColor(colors[self.ball_type])
        self.node.addTorque([(random()-0.5)/30,0,(random()-0.5)/30],True)
        self.label = names[self.ball_type]
        self.value = points[self.ball_type]
        self.alive = sec_alive*1000 + int((random()-0.5)*2000)

    def step(self, timestep, robo_pos)->int:
        points = 0
        if len(robo_pos)==3:
            a = np.asarray(self.node.getField("translation").getSFVec3f())
            b = np.asarray(robo_pos)
            mse = ((a - b)**2).mean()
            if mse < 0.005:
                points = self.value
                self.alive = 0
        self.alive -= timestep
        if self.alive <= 0:
            self.node.remove()
        return points
            


balls = [Ball()]

# Main loop:
# - perform simulation steps until Webots is stopping the controller

total_points = 0
last_points = 0

while sup.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()
    # Process sensor data here.
    if len(balls) < 6 and random() > 0.92:
        balls.append(Ball())

    

    #check for (almost) collision between robot and balls
    for ball in balls:
        reward = ball.step(timestep,robo.getPosition())
        total_points += reward
        if reward != 0:
            msg = {"event":"ball_collected", "value":reward, "ball_color": names[points.index(reward)]}
            emit.send(msgpack.packb(msg))
    balls = [ball for ball in balls if ball.alive > 0]
    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    diff = total_points-last_points
    diff_mod = "+" if diff > 0 else ""
    if diff != 0:
        #print(f"POINTS changed ({diff_mod}{diff}) -> {total_points}")
        last_points = total_points


# Enter here exit cleanup code.

"""my_controller1 controller."""

#https://cyberbotics.com/doc/guide/running-extern-robot-controllers?tab-language=python&tab-os=windows
# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from os import times
from controller import Robot, Motor, Camera, DistanceSensor

# create the Robot instance.
robot = Robot()
# get the time step of the current world.

#for debugging, print all devices on the robot

for i in range(robot.getNumberOfDevices()):
    print(str(robot.getDeviceByIndex(i)) + ": " + robot.getDeviceByIndex(i).getName())


print(robot.getName())

timestep = 128;#int(robot.getBasicTimeStep())


# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#get and reset motors
motorLeft:Motor = robot.getDevice('left wheel motor')
motorRight:Motor = robot.getDevice('right wheel motor')
motorLeft.setPosition(float('inf')) #this sets the motor to velocity control instead of position control
motorRight.setPosition(float('inf'))
motorLeft.setVelocity(0)
motorRight.setVelocity(0)

#get and enable some sensors
#sensorLeft:PositionSensor = robot.getDevice('left wheel sensor')
#sensorRight:PositionSensor = robot.getDevice('right wheel sensor')
#sensorLeft.enable(timestep)
#sensorRight.enable(timestep)

camera:Camera = robot.getDevice("camera")
camera.enable(timestep)
#this is constant
maxVelocity = motorLeft.getMaxVelocity()
#  ds = robot.getDistanceSensor('dsname')
#  ds.enable(timestep)


ls:DistanceSensor = robot.getDevice("dsLeft")
ls.enable(timestep)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()
    #img = camera.getImageArray()
    #npImg = np.asarray(img, dtype=np.uint8)
    #print(npImg.shape)
    print(ls.getValue())
    motorRight.setVelocity(1)
    # Process sensor data here.

    #speed = calcSpeed(keyPress,speed)

    # Enter here functions to send actuator commands, like:
    #motorLeft.setVelocity(maxVelocity*speed)
    #motorRight.setVelocity(maxVelocity*speed)
    #motorRight.setPosition(10.0)
    #print(sensorLeft.getValue())
    pass

print("closing")
# Enter here exit cleanup code.

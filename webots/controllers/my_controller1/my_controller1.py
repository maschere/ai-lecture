"""my_controller1 controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot

# create the Robot instance.
robot = Robot()

for i in range(robot.getNumberOfDevices()):
    print(str(robot.getDeviceByIndex(i)) + ": " + robot.getDeviceByIndex(i).getName())


# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())


motorLeft = robot.getDevice('left wheel motor')
motorRight = robot.getDevice('right wheel motor')
motorLeft.setPosition(float('inf')) #this sets the motor to velocity control instead of position control
motorRight.setPosition(float('inf'))
motorLeft.setVelocity(0)
motorRight.setVelocity(0)


ls = robot.getDevice("gsztrzu1")
ls.enable(timestep)

# You should insert a getDevice-like function in order to get the
# instance of a device of the robot. Something like:
#  motor = robot.getMotor('motorname')
#  ds = robot.getDistanceSensor('dsname')
#  ds.enable(timestep)

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1:
    # Read the sensors:
    # Enter here functions to read sensor data, like:
    #  val = ds.getValue()
    #motorLeft.setVelocity(1)
    print(ls.getValue())

    # Process sensor data here.

    # Enter here functions to send actuator commands, like:
    #  motor.setPosition(10.0)
    pass

# Enter here exit cleanup code.

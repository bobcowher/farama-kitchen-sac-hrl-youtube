import numpy as np
import pygame

class Controller:

    def __init__(self):
        self.gripper_closed = None

        pygame.init()
        pygame.joystick.init()

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

    def get_action(self):

        action = np.zeros(9)

        gripper_button_pressed = False

        # Map left joystick to joint1 and joint2 angular velocity
        action[0] = self.joystick.get_axis(0) # Left stick horizontal
        action[1] = self.joystick.get_axis(1) # Left stick vertical

        action[0] = action[0] * -1
        action[1] = action[1] * -1

        # Map right joystick to joints 3 and 4
        action[2] = self.joystick.get_axis(3)
        action[3] = self.joystick.get_axis(2)
        action[3] = action[3] * -1

        if self.joystick.get_button(0):
            action[4] = -1
            print("Button 0 pressed")
        elif self.joystick.get_button(2): 
            action[4] = 1
            print("Button 2 pressed")
        elif self.joystick.get_button(1):
            self.gripper_closed = True
            gripper_button_pressed = True
            print("Button 1 pressed")
        elif self.joystick.get_button(3):
            self.gripper_closed = False
            gripper_button_pressed = True
        elif self.joystick.get_button(4):
            action[5] = 1
            print("Button 4 pressed")
        elif self.joystick.get_button(5):
            action[5] = -1
            print("Button 5 pressed")
        elif self.joystick.get_button(6):
            action[6] = -1
            print("Button 6 pressed")
        elif self.joystick.get_button(7):
            action[6] = 1
            print("Button 7 pressed")
        elif self.joystick.get_button(8):
            action[7] = 1
            print("Button 8 pressed")
        elif self.joystick.get_button(9):
            action[7] = -1
            print("Button 9 pressed")
        
        mask = np.abs(action) >= 0.1
        action = action * mask
        action = np.where(action == -0.0, 0.0, action)

        if np.all(action == 0) and gripper_button_pressed == False:
            action = None
        else:
            if self.gripper_closed == True:
                action[7] = -1.0
                action[8] = -1.0
            elif self.gripper_closed == False:
                action[7] = 1.0
                action[8] = 1.0

        return action
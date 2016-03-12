import pygame
import random

# Define some colors
BLACK    = (   0,   0,   0)
WHITE    = ( 255, 255, 255)
# This is a simple class that will help us print to the screen
# It has nothing to do with the joysticks, just outputing the
# information.
class TextPrint:
    def __init__(self, screen):
        self.reset()
        self.screen = screen
        self.font = pygame.font.Font(None, 20)

    def text(self, textString):
        textBitmap = self.font.render(textString, True, BLACK)
        self.screen.blit(textBitmap, [self.x, self.y])
        self.y += self.line_height
        
    def reset(self):
        self.x = 10
        self.y = 10
        self.line_height = 15
        
    def indent(self):
        self.x += 10
        
    def unindent(self):
        self.x -= 10

class Stats():
    def __init__(self, screen):
        self.screen = screen
        self.textPrint=TextPrint(self.screen)
    def update(self, state):
        self.screen.fill(WHITE)
        self.textPrint.reset()
        self.textPrint.text(str(random.random()))
        pygame.display.flip()
        
        


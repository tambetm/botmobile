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
        self.font = pygame.font.Font(None, 30)

    def text(self, textString):
        textBitmap = self.font.render(textString, True, BLACK)
        self.screen.blit(textBitmap, [self.x, self.y])
        self.y += self.line_height
        
    def reset(self):
        self.x = 10
        self.y = 10
        self.line_height = 20
        
    def indent(self):
        self.x += 10
        
    def unindent(self):
        self.x -= 10

class Stats():
    def __init__(self, screen, inevery):
        """ updates screen only in inevery step"""
        self.screen = screen
        self.textPrint=TextPrint(self.screen)
        self.inevery = 10
        self.curstep = -1

    def update(self, state):
        self.curstep = (self.curstep + 1) % self.inevery
        if not self.curstep == 0:
            return
        self.screen.fill(WHITE)
        self.textPrint.reset()
        # angle
        self.textPrint.text(str(random.random()))
        self.textPrint.text('angle: {}'.format(state.angle))
        self.textPrint.text('curLapTime{}:'.format(state.curLapTime))
        self.textPrint.text('distFromStart: {}'.format(state.distFromStart))
        self.textPrint.text('distRaced :{}'.format(state.distRaced))
        self.textPrint.text('gear: {}'.format(state.gear))
        self.textPrint.text('lastLapTime: {}'.format(state.lastLapTime))
        self.textPrint.text('rpm: {}'.format(state.rpm))
        # speedx, y, z
        self.textPrint.text('speeds: X, Y, Z')
        self.textPrint.indent()
        self.textPrint.text('{}, {}, {}'.format(state.speedX, state.speedY, state.speedZ))

        self.textPrint.text('Trackpos: {}'.format(state.trackPos))
        # Tracks
        self.textPrint.text('Tracks: ')
        self.textPrint.indent()
        for i,sens in enumerate(state.track):
            sens_txt = str(-90 + i * 10) + ': ' + str(sens)
            self.textPrint.text(sens_txt)

        pygame.display.flip()
        
        


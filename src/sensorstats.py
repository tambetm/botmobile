import pygame
import random
import math

# Define some colors
BLACK    = (   0,   0,   0)
WHITE    = ( 255, 255, 255)
BARCOL   =(102, 170, 255)
LINECOL  =(0, 255,0)

# This is a simple class that will help us print to the screen
# It has nothing to do with the joysticks, just outputing the
# information.
class TextPrint:
    def __init__(self, screen):
        self.reset()
        self.screen = screen
        #print pygame.font.get_fonts()
        #print pygame.font.match_font('arial')
        self.font = pygame.font.Font(None, 30)
        self.centerx = screen.get_rect().centerx

    def text(self, textString):
        textBitmap = self.font.render(textString, True, BLACK)
        textpos = self.centerx - textBitmap.get_rect().width / 2.
        self.screen.blit(textBitmap, [textpos, self.y])
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

    def __init__(self, inevery):
        """ updates screen only in inevery step"""
        pygame.init() # needed for keyboard input
        self.screen = pygame.display.set_mode([300, 700])
        self.textPrint=TextPrint(self.screen)
        self.inevery = 10
        self.curstep = -1
        self.centerx = self.screen.get_rect().centerx 


    def addRect(self, centerx, y, progress, mancontrol):
        fontsize=  100
        font = pygame.font.Font(None, fontsize)
        rect_width = 150
        rect_height = 25
        percentage = int(progress*100)
       
        firstlnpos = y - 30
        barpos = (self.centerx - rect_width / 2, y+ fontsize -16)

        ### First line 
        self.textPrint.y = firstlnpos
        #if self.sta
        self.textPrint.text('Computer controls')
        #else:
       #     self.textPrint.text('Out of track')
       
        ## second lien percentage
        txt = font.render(str(percentage) + '%', True, (255,0,0))
        self.screen.blit(txt, [self.centerx + rect_width / 2 - 10- txt.get_rect().width, y])
       
        ## third line bar or text
        #if percentage < 100:
            # draw prog bar
        self.rect = pygame.draw.rect(self.screen, BARCOL, (barpos[0], barpos[1], rect_width*progress, rect_height))
        self.rect = pygame.draw.rect(self.screen, (BLACK), (self.centerx - rect_width / 2, y +fontsize -16, rect_width, rect_height), 2)

        
    def draw_tracks(self, track, y):
        rect_width = 50
        rect_height = 90
        trcenter = (self.centerx, y)
        for i, sens in enumerate(track):
            alpha = math.radians(-180 + i * 10)
            x = trcenter[0] + math.cos(alpha) * (sens  + 20) * 2
            y = trcenter[1] + math.sin(alpha) * (sens + 20) * 2
            pygame.draw.line(self.screen, LINECOL, trcenter, (x,y), 2)
        self.rect = pygame.draw.rect(self.screen, (BLACK), (self.centerx - rect_width / 2, y, rect_width, rect_height), 2)
    
    def draw_texts(self, state):
        # angle
        self.textPrint.text('bot conrol: {}'.format(state.botcontrol))
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
        self.textPrint.unindent()

        self.textPrint.text('Trackpos: {}'.format(state.trackPos))
        # Tracks
        self.textPrint.text('Tracks: ')
        self.textPrint.indent()
        for i,sens in enumerate(state.track):
            sens_txt = str(-90 + i * 10) + ': ' + str(sens)
            self.textPrint.text(sens_txt)



    def update(self, state):
        self.curstep = (self.curstep + 1) % self.inevery
        if not self.curstep == 0:
            return
        self.screen.fill(WHITE)
        self.textPrint.reset()
        self.addRect(self.centerx, 75, state.botcontrol, True)
        self.textPrint.y = 300
        #self.draw_texts(state)  
        self.draw_tracks(state.track, 450)
        #self.prog_bar.update(100)
        pygame.display.flip()

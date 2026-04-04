import os
import pygame
import math
import random


class BotFace:
    def __init__(self, width=400, height=400):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Gizmo")
        self.clock = pygame.time.Clock()

        self.state   = "idle"
        self.running = True

        # Colors
        self.BG_COLOR    = (30, 30, 40)
        self.FACE_COLOR  = (200, 200, 220)
        self.EYE_COLOR   = (40, 40, 50)
        self.MOUTH_COLOR = (40, 40, 50)
        self.BLUSH_COLOR = (255, 180, 180)

        # Animation
        self.blink_frame = 0
        self.is_blinking = False
        self.frame_count = 0
        self.bounce_y    = 0.0
        self.zzz_y       = 80.0

    def draw_face_base(self):
        breath = math.sin(self.frame_count * 0.05) * 3

        if self.state in ("happy", "surprised"):
            self.bounce_y = math.sin(self.frame_count * 0.4) * 8
        else:
            self.bounce_y = 0.0

        y = int(breath + self.bounce_y)
        pygame.draw.rect(
            self.screen, self.FACE_COLOR,
            pygame.Rect(50, 60 + y, 300, 280),
            border_radius=40
        )
        return y

    def draw_eyes(self, y):
        lx, ly = 150, 170 + y
        rx, ry = 250, 170 + y

        # ── Sleeping ──
        if self.state == "sleeping":
            pygame.draw.line(self.screen, self.EYE_COLOR, (130, ly), (170, ly), 4)
            pygame.draw.line(self.screen, self.EYE_COLOR, (230, ry), (270, ry), 4)
            return

        # ── Drowsy ──
        if self.state == "drowsy":
            pygame.draw.ellipse(self.screen, self.EYE_COLOR,  pygame.Rect(lx-20, ly-25, 40, 50))
            pygame.draw.ellipse(self.screen, self.EYE_COLOR,  pygame.Rect(rx-20, ry-25, 40, 50))
            # Heavy eyelid covers top half
            pygame.draw.rect(self.screen, self.FACE_COLOR, pygame.Rect(lx-22, ly-28, 44, 28))
            pygame.draw.rect(self.screen, self.FACE_COLOR, pygame.Rect(rx-22, ry-28, 44, 28))
            return

        # ── Happy ^^  ──
        if self.state == "happy":
            pygame.draw.arc(self.screen, self.EYE_COLOR, pygame.Rect(130, 160+y, 40, 30), 0, math.pi, 5)
            pygame.draw.arc(self.screen, self.EYE_COLOR, pygame.Rect(230, 160+y, 40, 30), 0, math.pi, 5)
            pygame.draw.circle(self.screen, self.BLUSH_COLOR, (110, 220+y), 15)
            pygame.draw.circle(self.screen, self.BLUSH_COLOR, (290, 220+y), 15)
            return

        # ── Surprised ──
        if self.state == "surprised":
            pygame.draw.circle(self.screen, self.EYE_COLOR, (lx, ly), 25)
            pygame.draw.circle(self.screen, self.EYE_COLOR, (rx, ry), 25)
            return

        # ── Blink logic (idle / talking / thinking / bored) ──
        eye_h = 50
        if not self.is_blinking and random.random() < 0.01:
            self.is_blinking = True
            self.blink_frame = 0

        if self.is_blinking:
            self.blink_frame += 1
            if self.blink_frame <= 5:
                eye_h = max(2, 50 - self.blink_frame * 10)
            elif self.blink_frame <= 10:
                eye_h = max(2, 2 + (self.blink_frame - 5) * 10)
            else:
                self.is_blinking = False
                eye_h = 50

        # ── Bored: right eye droopy ──
        if self.state == "bored":
            pygame.draw.ellipse(self.screen, self.EYE_COLOR, pygame.Rect(lx-20, ly-eye_h//2,    40, eye_h))
            pygame.draw.ellipse(self.screen, self.EYE_COLOR, pygame.Rect(rx-20, ry-eye_h//4,    40, eye_h//2))
            return

        # ── Default ──
        pygame.draw.ellipse(self.screen, self.EYE_COLOR, pygame.Rect(lx-20, ly-eye_h//2, 40, eye_h))
        pygame.draw.ellipse(self.screen, self.EYE_COLOR, pygame.Rect(rx-20, ry-eye_h//2, 40, eye_h))

    def draw_mouth(self, y):
        cx, cy = 200, 260 + y

        if self.state == "talking":
            h = int(15 + abs(math.sin(self.frame_count * 0.6)) * 25)
            w = int(50 + math.cos(self.frame_count * 0.3) * 10)
            pygame.draw.ellipse(self.screen, self.MOUTH_COLOR,
                pygame.Rect(cx - w//2, cy - h//2, w, h))

        elif self.state == "happy":
            pygame.draw.arc(self.screen, self.MOUTH_COLOR,
                pygame.Rect(160, 230+y, 80, 60), math.pi, 0, 5)

        elif self.state == "thinking":
            points = [(x, int(265 + y + math.sin(x * 0.2 + self.frame_count * 0.2) * 4))
                      for x in range(170, 230, 5)]
            if len(points) >= 2:
                pygame.draw.lines(self.screen, self.MOUTH_COLOR, False, points, 3)

        elif self.state == "surprised":
            pygame.draw.circle(self.screen, self.MOUTH_COLOR, (cx, cy), 15)

        elif self.state == "sleeping":
            pygame.draw.ellipse(self.screen, self.MOUTH_COLOR,
                pygame.Rect(185, 255+y, 30, 12))

        elif self.state == "drowsy":
            pygame.draw.line(self.screen, self.MOUTH_COLOR,
                (175, 260+y), (225, 260+y), 3)

        elif self.state == "bored":
            pygame.draw.arc(self.screen, self.MOUTH_COLOR,
                pygame.Rect(175, 250+y, 50, 30), 0, math.pi, 3)

        else:  # idle / default smile
            pygame.draw.arc(self.screen, self.MOUTH_COLOR,
                pygame.Rect(175, 240+y, 50, 40), math.pi, 0, 3)

    def draw_zzz(self):
        font_big   = pygame.font.SysFont("arial", 28, bold=True)
        font_small = pygame.font.SysFont("arial", 18, bold=True)

        self.zzz_y -= 0.4
        if self.zzz_y < 20:
            self.zzz_y = 80.0

        zy = int(self.zzz_y)
        self.screen.blit(font_big.render("z",   True, (120, 140, 200)), (270, zy))
        self.screen.blit(font_small.render("z", True, (100, 120, 180)), (290, zy + 22))
        self.screen.blit(font_small.render("z", True, (80,  100, 160)), (308, zy + 38))

    def update(self):
        self.frame_count += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        # Background
        if self.state == "sleeping":
            self.screen.fill((15, 15, 30))
        elif self.state == "drowsy":
            self.screen.fill((20, 22, 35))
        else:
            self.screen.fill(self.BG_COLOR)

        y = self.draw_face_base()
        self.draw_eyes(y)
        self.draw_mouth(y)

        if self.state == "sleeping":
            self.draw_zzz()

        pygame.display.flip()
        self.clock.tick(30)
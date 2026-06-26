import math
import random
import time
import tkinter as tk
from dataclasses import dataclass, field


WIDTH = 520
HEIGHT = 760
FPS_MS = 16


def clamp(value, low, high):
    return max(low, min(high, value))


def overlap(a, b):
    return (
        abs(a.x - b.x) * 2 < (a.w + b.w)
        and abs(a.y - b.y) * 2 < (a.h + b.h)
    )


@dataclass
class Entity:
    x: float
    y: float
    w: float
    h: float
    hp: int
    vx: float = 0
    vy: float = 0
    kind: str = ""
    cooldown: int = 0
    alive: bool = True
    data: dict = field(default_factory=dict)

    def hit(self, damage):
        self.hp -= damage
        if self.hp <= 0:
            self.alive = False


@dataclass
class Bullet:
    x: float
    y: float
    r: float
    damage: int
    vx: float
    vy: float
    owner: str
    color: str
    alive: bool = True

    @property
    def w(self):
        return self.r * 2

    @property
    def h(self):
        return self.r * 2


@dataclass
class Drop:
    x: float
    y: float
    kind: str
    vy: float = 2.2
    alive: bool = True

    @property
    def w(self):
        return 24

    @property
    def h(self):
        return 24


class AirplaneWar:
    def __init__(self, root):
        self.root = root
        self.root.title("飞机大战 - Boss / 关卡 / 掉落 / 武器升级")
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="#07111f", highlightthickness=0)
        self.canvas.pack()

        self.keys = set()
        self.root.bind("<KeyPress>", self.key_down)
        self.root.bind("<KeyRelease>", self.key_up)

        self.reset()
        self.loop()

    def reset(self):
        self.player = Entity(WIDTH / 2, HEIGHT - 90, 40, 46, hp=100, kind="player")
        self.bullets = []
        self.enemy_bullets = []
        self.enemies = []
        self.drops = []
        self.particles = []
        self.level = 1
        self.score = 0
        self.weapon_level = 1
        self.spread_level = 0
        self.shield_until = 0
        self.laser_until = 0
        self.boss = None
        self.frame = 0
        self.level_kills = 0
        self.spawn_timer = 0
        self.message = "方向键/WASD移动，空格射击，P暂停"
        self.message_until = self.now() + 3
        self.paused = False
        self.game_over = False
        self.win = False

    def now(self):
        return time.monotonic()

    def key_down(self, event):
        key = event.keysym.lower()
        if key == "r" and self.game_over:
            self.reset()
            return
        if key == "p":
            self.paused = not self.paused
            return
        self.keys.add(key)

    def key_up(self, event):
        self.keys.discard(event.keysym.lower())

    def loop(self):
        if not self.paused and not self.game_over:
            self.update()
        self.draw()
        self.root.after(FPS_MS, self.loop)

    def update(self):
        self.frame += 1
        self.move_player()
        self.spawn_enemies()
        self.update_bullets()
        self.update_enemies()
        self.update_drops()
        self.handle_collisions()
        self.cleanup()
        self.check_level_progress()

    def move_player(self):
        speed = 5.0
        dx = dy = 0
        if "left" in self.keys or "a" in self.keys:
            dx -= speed
        if "right" in self.keys or "d" in self.keys:
            dx += speed
        if "up" in self.keys or "w" in self.keys:
            dy -= speed
        if "down" in self.keys or "s" in self.keys:
            dy += speed
        if dx and dy:
            dx *= 0.72
            dy *= 0.72
        self.player.x = clamp(self.player.x + dx, self.player.w / 2, WIDTH - self.player.w / 2)
        self.player.y = clamp(self.player.y + dy, self.player.h / 2 + 52, HEIGHT - self.player.h / 2)

        fire_rate = max(8, 17 - self.weapon_level * 2)
        if ("space" in self.keys or "j" in self.keys) and self.frame % fire_rate == 0:
            self.fire_player()
        if self.now() < self.laser_until and self.frame % 3 == 0:
            self.fire_laser()

    def fire_player(self):
        damage = 9 + self.weapon_level * 3
        lanes = [0]
        if self.weapon_level >= 2:
            lanes = [-10, 10]
        if self.weapon_level >= 4:
            lanes = [-18, 0, 18]
        for lane in lanes:
            self.bullets.append(Bullet(self.player.x + lane, self.player.y - 28, 4, damage, lane * 0.04, -9, "player", "#8ffcff"))
        if self.spread_level:
            for angle in range(1, self.spread_level + 1):
                vx = 1.25 * angle
                self.bullets.append(Bullet(self.player.x, self.player.y - 20, 3, damage - 2, -vx, -8, "player", "#fbd38d"))
                self.bullets.append(Bullet(self.player.x, self.player.y - 20, 3, damage - 2, vx, -8, "player", "#fbd38d"))

    def fire_laser(self):
        self.bullets.append(Bullet(self.player.x, self.player.y - 120, 7, 16 + self.weapon_level * 2, 0, -14, "player", "#b6ff6a"))

    def spawn_enemies(self):
        if self.boss:
            return
        self.spawn_timer -= 1
        if self.spawn_timer > 0:
            return
        self.spawn_timer = max(18, 55 - self.level * 5)
        roll = random.random()
        if roll < 0.18 + self.level * 0.02:
            enemy = Entity(random.randint(40, WIDTH - 40), -30, 50, 42, 34 + self.level * 8, vy=1.2 + self.level * 0.25, kind="elite")
        else:
            enemy = Entity(random.randint(28, WIDTH - 28), -30, 34, 32, 18 + self.level * 4, vy=2.0 + self.level * 0.28, kind="normal")
        enemy.vx = random.choice([-1, 1]) * random.uniform(0.4, 1.3)
        enemy.cooldown = random.randint(35, 90)
        self.enemies.append(enemy)

    def update_bullets(self):
        for bullet in self.bullets + self.enemy_bullets:
            bullet.x += bullet.vx
            bullet.y += bullet.vy
            if bullet.y < -30 or bullet.y > HEIGHT + 30 or bullet.x < -30 or bullet.x > WIDTH + 30:
                bullet.alive = False

    def update_enemies(self):
        for enemy in self.enemies:
            enemy.x += enemy.vx
            enemy.y += enemy.vy
            if enemy.x < enemy.w / 2 or enemy.x > WIDTH - enemy.w / 2:
                enemy.vx *= -1
            enemy.cooldown -= 1
            if enemy.cooldown <= 0 and enemy.y > 10:
                enemy.cooldown = random.randint(65, 115)
                self.enemy_bullets.append(Bullet(enemy.x, enemy.y + 20, 5, 9 + self.level, 0, 4.2, "enemy", "#ff7070"))
            if enemy.y > HEIGHT + 50:
                enemy.alive = False

        if self.boss:
            self.update_boss()

    def update_boss(self):
        boss = self.boss
        if boss.y < 105:
            boss.y += 1.2
        boss.x += boss.vx
        if boss.x < 88 or boss.x > WIDTH - 88:
            boss.vx *= -1
        boss.cooldown -= 1
        if boss.cooldown <= 0:
            boss.cooldown = max(18, 42 - self.level * 2)
            for i in range(-2, 3):
                self.enemy_bullets.append(Bullet(boss.x + i * 24, boss.y + 38, 5, 12 + self.level, i * 0.75, 4.1, "enemy", "#ff4d8d"))
            if self.frame % 120 < 45:
                for angle in range(0, 360, 36):
                    rad = math.radians(angle)
                    self.enemy_bullets.append(Bullet(boss.x, boss.y, 4, 8 + self.level, math.cos(rad) * 2.4, math.sin(rad) * 2.4, "enemy", "#ffbd59"))

    def update_drops(self):
        for drop in self.drops:
            drop.y += drop.vy
            if drop.y > HEIGHT + 30:
                drop.alive = False

    def handle_collisions(self):
        targets = self.enemies + ([self.boss] if self.boss else [])
        for bullet in self.bullets:
            if not bullet.alive:
                continue
            for enemy in targets:
                if enemy and enemy.alive and overlap(bullet, enemy):
                    enemy.hit(bullet.damage)
                    bullet.alive = False
                    self.add_spark(bullet.x, bullet.y, "#fff6a6")
                    if not enemy.alive:
                        self.kill_enemy(enemy)
                    break

        shielded = self.now() < self.shield_until
        for bullet in self.enemy_bullets:
            if bullet.alive and overlap(bullet, self.player):
                bullet.alive = False
                if not shielded:
                    self.player.hit(bullet.damage)
                self.add_spark(bullet.x, bullet.y, "#ff8c8c")

        for enemy in self.enemies:
            if enemy.alive and overlap(enemy, self.player):
                enemy.alive = False
                if not shielded:
                    self.player.hit(18)
                self.add_spark(enemy.x, enemy.y, "#ff8c8c")

        if self.boss and self.boss.alive and overlap(self.boss, self.player):
            if not shielded and self.frame % 20 == 0:
                self.player.hit(16)

        for drop in self.drops:
            if drop.alive and overlap(drop, self.player):
                drop.alive = False
                self.apply_drop(drop.kind)

        if self.player.hp <= 0:
            self.game_over = True
            self.message = "游戏结束，按 R 重新开始"
            self.message_until = self.now() + 60

    def kill_enemy(self, enemy):
        if enemy is self.boss:
            self.score += 500 * self.level
            self.drop_items(enemy.x, enemy.y, boss=True)
            self.boss = None
            self.level += 1
            self.level_kills = 0
            self.player.hp = min(100, self.player.hp + 25)
            if self.level > 5:
                self.win = True
                self.game_over = True
                self.message = "通关成功！按 R 再来一局"
            else:
                self.message = f"第 {self.level} 关开始"
            self.message_until = self.now() + 3
            return

        self.level_kills += 1
        self.score += 40 if enemy.kind == "normal" else 90
        self.drop_items(enemy.x, enemy.y, boss=False)

    def drop_items(self, x, y, boss=False):
        chance = 1.0 if boss else (0.22 if random.random() < 0.55 else 0)
        if not chance:
            return
        kinds = ["power", "heal", "shield", "spread", "laser"]
        count = 3 if boss else 1
        for i in range(count):
            kind = random.choices(kinds, weights=[35, 20, 18, 16, 11], k=1)[0]
            self.drops.append(Drop(x + (i - count // 2) * 28, y, kind))

    def apply_drop(self, kind):
        names = {
            "power": "火力升级",
            "heal": "维修包",
            "shield": "护盾",
            "spread": "散射插件",
            "laser": "激光炮",
        }
        if kind == "power":
            self.weapon_level = min(5, self.weapon_level + 1)
        elif kind == "heal":
            self.player.hp = min(100, self.player.hp + 24)
        elif kind == "shield":
            self.shield_until = self.now() + 6
        elif kind == "spread":
            self.spread_level = min(3, self.spread_level + 1)
        elif kind == "laser":
            self.laser_until = self.now() + 7
        self.message = names[kind]
        self.message_until = self.now() + 1.2

    def check_level_progress(self):
        if self.boss or self.game_over:
            return
        needed = 10 + self.level * 4
        if self.level_kills >= needed:
            hp = 230 + self.level * 115
            self.boss = Entity(WIDTH / 2, -80, 150, 86, hp=hp, vx=2.2 + self.level * 0.15, kind="boss")
            self.boss.data["max_hp"] = hp
            self.boss.cooldown = 50
            self.message = f"Boss 来袭：第 {self.level} 关"
            self.message_until = self.now() + 3

    def cleanup(self):
        self.bullets = [b for b in self.bullets if b.alive]
        self.enemy_bullets = [b for b in self.enemy_bullets if b.alive]
        self.enemies = [e for e in self.enemies if e.alive]
        self.drops = [d for d in self.drops if d.alive]
        self.particles = [(x, y, c, life - 1) for x, y, c, life in self.particles if life > 1]

    def add_spark(self, x, y, color):
        for _ in range(5):
            self.particles.append((x + random.randint(-8, 8), y + random.randint(-8, 8), color, random.randint(8, 16)))

    def draw(self):
        self.canvas.delete("all")
        self.draw_background()
        self.draw_player()
        for enemy in self.enemies:
            self.draw_enemy(enemy)
        if self.boss:
            self.draw_boss()
        for bullet in self.bullets + self.enemy_bullets:
            self.canvas.create_oval(bullet.x - bullet.r, bullet.y - bullet.r, bullet.x + bullet.r, bullet.y + bullet.r, fill=bullet.color, outline="")
        for drop in self.drops:
            self.draw_drop(drop)
        for x, y, color, life in self.particles:
            size = max(1, life // 3)
            self.canvas.create_oval(x - size, y - size, x + size, y + size, fill=color, outline="")
        self.draw_hud()

    def draw_background(self):
        for i in range(38):
            y = (i * 67 + self.frame * (1 + i % 3)) % HEIGHT
            x = (i * 89 + 37) % WIDTH
            color = "#335070" if i % 4 else "#557aa3"
            self.canvas.create_rectangle(x, y, x + 2, y + 6, fill=color, outline="")

    def draw_player(self):
        x, y = self.player.x, self.player.y
        color = "#53d8fb" if self.now() >= self.shield_until else "#8fffb6"
        points = [x, y - 25, x - 22, y + 22, x, y + 12, x + 22, y + 22]
        self.canvas.create_polygon(points, fill=color, outline="#d9ffff", width=2)
        self.canvas.create_oval(x - 7, y - 6, x + 7, y + 10, fill="#163855", outline="")
        if self.now() < self.shield_until:
            self.canvas.create_oval(x - 34, y - 36, x + 34, y + 36, outline="#8fffb6", width=3)

    def draw_enemy(self, enemy):
        x, y = enemy.x, enemy.y
        fill = "#c44dff" if enemy.kind == "elite" else "#ff725c"
        self.canvas.create_polygon(x, y + 19, x - enemy.w / 2, y - 8, x - 10, y - 16, x, y - 8, x + 10, y - 16, x + enemy.w / 2, y - 8, fill=fill, outline="#ffd2cc")

    def draw_boss(self):
        boss = self.boss
        x, y = boss.x, boss.y
        self.canvas.create_rectangle(x - 76, y - 32, x + 76, y + 34, fill="#7b2cff", outline="#f2dcff", width=2)
        self.canvas.create_polygon(x - 98, y + 5, x - 76, y - 25, x - 76, y + 30, fill="#a14dff", outline="")
        self.canvas.create_polygon(x + 98, y + 5, x + 76, y - 25, x + 76, y + 30, fill="#a14dff", outline="")
        self.canvas.create_oval(x - 23, y - 20, x + 23, y + 24, fill="#250c48", outline="#ffa8ff")
        ratio = max(0, boss.hp / boss.data.get("max_hp", boss.hp))
        self.canvas.create_rectangle(78, 44, WIDTH - 78, 58, fill="#35102e", outline="#ff9cd9")
        self.canvas.create_rectangle(80, 46, 80 + (WIDTH - 160) * ratio, 56, fill="#ff4d8d", outline="")
        self.canvas.create_text(WIDTH / 2, 30, text=f"BOSS HP {max(0, boss.hp)}", fill="#ffd6ef", font=("Arial", 12, "bold"))

    def draw_drop(self, drop):
        colors = {
            "power": "#ffe066",
            "heal": "#69db7c",
            "shield": "#74c0fc",
            "spread": "#ffa94d",
            "laser": "#b2f2bb",
        }
        labels = {
            "power": "P",
            "heal": "+",
            "shield": "S",
            "spread": "W",
            "laser": "L",
        }
        self.canvas.create_rectangle(drop.x - 12, drop.y - 12, drop.x + 12, drop.y + 12, fill=colors[drop.kind], outline="#ffffff")
        self.canvas.create_text(drop.x, drop.y, text=labels[drop.kind], fill="#111111", font=("Arial", 11, "bold"))

    def draw_hud(self):
        self.canvas.create_rectangle(0, 0, WIDTH, 42, fill="#0b1729", outline="")
        self.canvas.create_text(12, 14, anchor="w", text=f"关卡 {self.level}/5", fill="#e6f4ff", font=("Arial", 12, "bold"))
        self.canvas.create_text(96, 14, anchor="w", text=f"分数 {self.score}", fill="#e6f4ff", font=("Arial", 12, "bold"))
        self.canvas.create_text(200, 14, anchor="w", text=f"武器 Lv.{self.weapon_level}", fill="#e6f4ff", font=("Arial", 12, "bold"))
        self.canvas.create_text(312, 14, anchor="w", text=f"散射 {self.spread_level}", fill="#e6f4ff", font=("Arial", 12, "bold"))
        self.canvas.create_rectangle(12, 25, 168, 35, fill="#35131b", outline="#ffffff")
        self.canvas.create_rectangle(14, 27, 14 + 152 * max(0, self.player.hp / 100), 33, fill="#ff5d73", outline="")
        if self.now() < self.message_until:
            self.canvas.create_text(WIDTH / 2, 76, text=self.message, fill="#fff4b8", font=("Microsoft YaHei", 16, "bold"))
        if self.paused:
            self.canvas.create_text(WIDTH / 2, HEIGHT / 2, text="暂停", fill="#ffffff", font=("Microsoft YaHei", 30, "bold"))
        if self.game_over:
            text = "通关成功！按 R 重新开始" if self.win else "游戏结束，按 R 重新开始"
            self.canvas.create_rectangle(55, HEIGHT / 2 - 65, WIDTH - 55, HEIGHT / 2 + 65, fill="#0b1729", outline="#e6f4ff")
            self.canvas.create_text(WIDTH / 2, HEIGHT / 2 - 15, text=text, fill="#ffffff", font=("Microsoft YaHei", 22, "bold"))
            self.canvas.create_text(WIDTH / 2, HEIGHT / 2 + 25, text=f"最终分数：{self.score}", fill="#ffe066", font=("Arial", 16, "bold"))


def main():
    root = tk.Tk()
    AirplaneWar(root)
    root.mainloop()


if __name__ == "__main__":
    main()

import random                 # случайные действия (исследование) и выборка из буфера
import shutil                 # резервная копия low-speed best
import sys                    # флаги --play / --train
import threading              # 4 окна = 4 потока + общий доступ к сети и буферу
import time                   # паузы между решениями и таймауты
from collections import deque # буфер воспроизведения фиксированной длины
from pathlib import Path      # путь к файлу модели независимо от текущей папки

import numpy as np                 # вектор состояния и стек батча
import torch                       # сама сеть и автоградиент
import torch.nn as nn              # слои (Linear), clip_grad_norm_
import torch.nn.functional as F    # ReLU и smooth_l1_loss
from playwright.sync_api import sync_playwright  # управление Chrome

GAME_URL = "https://wayou.github.io/t-rex-runner/"

# "chrome" - Chrome / Chrome for Testing, None - Chromium из Playwright
BROWSER_CHANNEL = "chrome"

# ---------------- режим работы ----------------
# True  - финальный забег: 1 окно, best_dqn.pt, без обучения и случайности
# False - обучение
# Переключать здесь или флагами: --play / --train
PLAY_BEST = True
if "--train" in sys.argv:
    PLAY_BEST = False
elif "--play" in sys.argv:
    PLAY_BEST = True

# False - обучение с исследованием (epsilon) и проверками рекорда.
# Это ближе к исходной Q-learning презентации, где EPSILON = 0.1.
TRAIN_GREEDY = False

# ---------------- параметры ----------------
FINAL_ATTEMPTS = 5        # финал: несколько попыток, берём лучшую
WINDOWS = 1 if PLAY_BEST else 4   # сколько окон Chrome одновременно
HEADLESS = False if PLAY_BEST else True  # финал - видно окно, обучение - без окон
EPISODES = FINAL_ATTEMPTS if PLAY_BEST else 1000  # всего эпизодов на все окна
TIME_LIMIT_MINUTES = 0 if PLAY_BEST else 120  # остановка по времени (0 = без лимита)
MAX_STEPS = 0             # 0 = без лимита шагов, эпизод до падения
MAX_EPISODE_SECONDS = 0 if (PLAY_BEST or TRAIN_GREEDY) else 300  # 0 = без лимита времени
STEP_DELAY = 0.02         # пауза между решениями: ~50 решений в секунду

GAMMA = 0.99              # дисконт: насколько важна будущая награда
LR = 1e-3                 # шаг оптимизатора Adam
BATCH = 64                # размер батча на одном шаге обучения
BUFFER_SIZE = 100_000     # сколько переходов хранить в памяти
TRAIN_EVERY = 4           # обучать раз в 4 решения
TARGET_SYNC = 1000        # как часто копировать online-сеть в target-сеть
HIDDEN = 128              # число нейронов в скрытом слое

DIST_REWARD = 0.01        # награда за единицу пройденной дистанции
PASS_REWARD = 1.0         # награда за каждое пройденное препятствие
CRASH_REWARD = -10.0      # штраф за падение

EPS_START = 0.0 if (PLAY_BEST or TRAIN_GREEDY) else 0.30  # доля случайных действий в начале
RESUME_EPS_START = 0.0 if TRAIN_GREEDY else 0.10  # старт исследования с best_dqn.pt
EPS_END = 0.01            # в конце почти строго по сети, 1% исследования
EPS_DECAY_DECISIONS = 500_000  # за сколько решений пройти от EPS_START к EPS_END

# учебная программа: (ограничение скорости, средняя дистанция для перехода)
CURRICULUM = [(6.5, 8000), (9.0, 20000), (13.0, None)]
FINAL_MAX_SPEED = 13.0    # финал: полная скорость, чтобы были птеродактили
TRAIN_STAGE = 2           # обучение сразу на MAX_SPEED=13.0

SAVE_EVERY = 20                     # сохранять модель каждые N эпизодов
RECORD_EVERY = 100                  # раз в N эпизодов проверять рекорд без исследования
RECORD_ATTEMPT_SECONDS = 120        # лимит проверочного забега (2 минуты)
METRIC_VERSION = 3                  # версия методики замера дистанции
MODEL_FILE = Path(__file__).resolve().parent / "dqn.pt"       # последние веса
BEST_FILE = Path(__file__).resolve().parent / "best_dqn.pt"   # лучшие веса
RESULTS_FILE = Path(__file__).resolve().parent / "results.csv"  # журнал эпизодов

# ---------------- действия ----------------
ACTION_RUN, ACTION_JUMP, ACTION_DUCK = 0, 1, 2  # коды действий
N_ACTIONS = 3                                    # сколько выходов у сети

# ---------------- классы препятствий ----------------
NO_OBSTACLE = 0     # препятствий нет
CACTUS_SMALL = 1    # кактус малый
CACTUS_LARGE = 2    # кактус большой
BIRD_LOW = 3        # птица у земли: надо прыгать
BIRD_MID = 4        # птица на средней высоте: надо присесть
BIRD_HIGH = 5       # птица высоко: можно бежать

N_INPUT = 12             # размер вектора состояния
DINO_FRONT_OFFSET = 40   # нос динозавра относительно tRex.xPos

# ---------------- правила прыжка ----------------
JUMP_MIN_FRAMES = 6      # раньше этого прыжок бесполезен (нос ещё далеко)
JUMP_MAX_FRAMES = 24     # позже этого прыжок уже не спасёт от препятствия
JUMPABLE = (CACTUS_SMALL, CACTUS_LARGE, BIRD_LOW)  # что перепрыгивается

torch.set_num_threads(2)  # не занимать все ядра: 4 окна и так грузят CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU, если есть

# ---------------- общие объекты потоков ----------------
stop_event = threading.Event()   # сигнал "остановиться" для всех окон
episode_lock = threading.Lock()  # защита счётчика эпизодов
results_lock = threading.Lock()  # защита записи в results.csv
episodes_started = 0             # сколько эпизодов уже начато
curriculum_lock = threading.Lock()  # защита номера этапа
curriculum_stage = 0                # текущий этап учебной программы


def log(message):
    """Печать с временем. flush=True нужен, чтобы строки не залипали в буфере."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


RESULTS_HEADER = "time,window,episode,distance,steps,reward,epsilon,stage,reason,best_distance\n"


def init_results_file():
    """Создать results.csv. Старый формат убрать в results_legacy.csv."""
    if RESULTS_FILE.exists():
        try:
            first_line = RESULTS_FILE.read_text(encoding="utf-8").splitlines()[0]
        except (OSError, IndexError):
            first_line = ""
        if first_line.strip() != RESULTS_HEADER.strip():
            legacy = RESULTS_FILE.with_name("results_legacy.csv")
            if legacy.exists():
                legacy.unlink()
            RESULTS_FILE.replace(legacy)
            log(f"старый журнал сохранён как {legacy.name}")
    if not RESULTS_FILE.exists():
        RESULTS_FILE.write_text(RESULTS_HEADER, encoding="utf-8")


def write_result(window_id, episode, result, epsilon, stage, best_distance):
    """Дописать строку об эпизоде в results.csv."""
    line = (
        f"{time.strftime('%Y-%m-%d %H:%M:%S')},{window_id},{episode},"
        f"{result['distance']},{result['steps']},{result['reward']:.2f},"
        f"{epsilon:.3f},{stage},{result['reason']},{best_distance}\n"
    )
    with results_lock:
        with RESULTS_FILE.open("a", encoding="utf-8") as file:
            file.write(line)


def current_max_speed():
    """Текущее ограничение скорости для учебной программы."""
    with curriculum_lock:
        return CURRICULUM[curriculum_stage][0]


def maybe_advance_curriculum(recent_average):
    """Перейти на следующий этап, если средняя дистанция выше порога."""
    global curriculum_stage
    with curriculum_lock:
        if curriculum_stage >= len(CURRICULUM) - 1:  # последний этап
            return False
        limit = CURRICULUM[curriculum_stage][1]      # порог текущего этапа
        if limit is None or recent_average < limit:  # ещё рано
            return False
        curriculum_stage += 1                        # следующий этап
        speed = CURRICULUM[curriculum_stage][0]
        stage = curriculum_stage + 1
    log(f"учебная программа: этап {stage}, MAX_SPEED={speed}")
    return True


# ---------------- чтение состояния из игры ----------------
STATE_JS = """
() => {
    const runner = window.Runner && window.Runner.instance_;
    if (!runner) return {status: "NO_RUNNER"};      // игра ещё не создана
    if (document.visibilityState !== "visible") return {status: "HIDDEN"};  // окно свёрнуто
    if (runner.crashed) return {status: "DEAD"};    // динозавр разбился
    if (!runner.playing) return {status: "NOT_PLAYING"};  // пауза

    const dino = runner.tRex;
    const list = runner.horizon.obstacles || [];
    const obstacles = [];
    for (let i = 0; i < list.length && obstacles.length < 2; i++) {
        const o = list[i];
        // пропускаем препятствия, которые уже полностью позади динозавра
        if (o.xPos + o.width <= dino.xPos - 20) continue;
        obstacles.push({x: o.xPos, w: o.width, y: o.yPos, t: o.typeConfig.type});
    }

    return {
        status: "OK",
        x: dino.xPos,                 // координата динозавра
        speed: runner.currentSpeed,   // текущая скорость игры
        jumping: dino.jumping,        // в прыжке?
        ducking: dino.ducking,        // присел?
        dist: runner.distanceRan,     // набранная дистанция
        passed: (runner.horizon.obstacles || []).filter(o => o.xPos + o.width < dino.xPos).length,  // сколько уже позади
        obstacles: obstacles          // до двух ближайших препятствий
    };
}
"""


def read_game(page):
    """Один вызов JS: получить состояние. При ошибке вернуть статус ERROR."""
    try:
        return page.evaluate(STATE_JS)
    except Exception as error:
        return {"status": "ERROR", "error": str(error)}


def page_visible(page):
    """Окно не свёрнуто: иначе rAF останавливается и эпизод недействителен."""
    try:
        return bool(page.evaluate("() => document.visibilityState === 'visible'"))
    except Exception:
        return False


# Отключить паузу игры на blur/focus/visibilitychange.
# Обработчики в игре подписаны через .bind(this), поэтому заменить свойство
# Runner.instance_.onVisibilityChange после загрузки нельзя - только не дать
# им подписаться вообще.
NO_PAUSE_JS = """
(() => {
    const add = EventTarget.prototype.addEventListener;
    EventTarget.prototype.addEventListener = function (type, listener, options) {
        if (type === 'blur' || type === 'focus' || type === 'visibilitychange') return;
        return add.call(this, type, listener, options);
    };
})();
"""

# Новый забег с нуля: restart() не сработает, пока идёт rAF, поэтому stop().
# startGame() включает arcade-mode и растягивает контейнер: без него виден
# только динозавр в полоске шириной 44 px.
RESTART_JS = """
() => {
    const runner = window.Runner && window.Runner.instance_;
    if (!runner) return false;
    runner.stop();
    runner.restart();
    runner.paused = false;
    runner.activated = true;        // пропустить вступление
    runner.playingIntro = false;
    runner.tRex.playingIntro = false;
    runner.containerEl.style.width = runner.dimensions.WIDTH + 'px';
    runner.containerEl.style.height = runner.dimensions.HEIGHT + 'px';
    const box = document.getElementById('messageBox');
    if (box) box.style.visibility = 'hidden';  // убрать "Press Space to start"
    runner.startGame();             // arcade-mode: масштаб и видимая ширина
    return true;
}
"""


def obstacle_class(obstacle):
    """Определить класс препятствия по типу и высоте птицы."""
    if obstacle["t"] == "PTERODACTYL":
        y = obstacle["y"]
        if y <= 60:      # высоко: пролетит над головой
            return BIRD_HIGH
        if y <= 92:      # средняя высота: только присед
            return BIRD_MID
        return BIRD_LOW  # у земли: только прыжок
    if obstacle["t"] == "CACTUS_SMALL":
        return CACTUS_SMALL
    return CACTUS_LARGE


def primary_obstacle(raw):
    """Первое препятствие, которое ещё впереди, а не только что пройдено."""
    obstacles = raw.get("obstacles") or []
    if not obstacles:
        return None
    speed = max(float(raw["speed"]), 6.0)
    front = raw["x"] + DINO_FRONT_OFFSET
    for obstacle in obstacles:
        if (obstacle["x"] - front) / speed >= 0.0:
            return obstacle
    return obstacles[-1]


def forced_bird_action(raw):
    """Правило высоты птеродактиля: низкий — прыжок, средний — присед,
    высокий — бег. Гарантирует правильное действие, не надеясь на сеть."""
    obstacle = primary_obstacle(raw)
    if obstacle is None:
        return None
    cls = obstacle_class(obstacle)
    if cls not in (BIRD_LOW, BIRD_MID, BIRD_HIGH):
        return None

    speed = max(float(raw["speed"]), 6.0)
    dt = (obstacle["x"] - (raw["x"] + DINO_FRONT_OFFSET)) / speed

    if cls == BIRD_LOW and 8.0 <= dt <= 16.0:
        return ACTION_JUMP
    if cls == BIRD_MID and 0.0 <= dt <= 20.0:
        return ACTION_DUCK
    if cls == BIRD_HIGH and 0.0 <= dt <= 20.0:
        return ACTION_RUN
    return None


def jump_allowed(raw):
    """Прыжок только у препятствия, которое действительно перепрыгивается.

    Между кактусами препятствий в зоне прыжка нет - динозавр бежит.
    Среднюю и высокую птицу прыжком не пройти: там нужен присед или бег.
    """
    obstacle = primary_obstacle(raw)
    if obstacle is None:
        return False
    speed = max(float(raw["speed"]), 6.0)
    dt = (obstacle["x"] - (raw["x"] + DINO_FRONT_OFFSET)) / speed
    return (
        JUMP_MIN_FRAMES <= dt <= JUMP_MAX_FRAMES
        and obstacle_class(obstacle) in JUMPABLE
    )


def encode(raw):
    """Состояние игры -> вектор из 12 чисел для сети."""
    obstacles = raw.get("obstacles") or []
    speed = max(float(raw["speed"]), 6.0)  # защита от деления на ноль
    cls = NO_OBSTACLE
    dt = 60.0      # по умолчанию "далеко"
    width = 0.0    # ширина препятствия в кадрах
    gap = 60.0     # зазор до второго препятствия в кадрах

    if obstacles:
        o = obstacles[0]
        cls = obstacle_class(o)
        # время до препятствия в кадрах = расстояние / скорость
        dt = (o["x"] - (raw["x"] + DINO_FRONT_OFFSET)) / speed
        width = o["w"] / speed
        if len(obstacles) > 1:
            # зазор между первым и вторым: нужно, чтобы не приземлиться между ними
            gap = (obstacles[1]["x"] - (o["x"] + o["w"])) / speed

    vec = np.zeros(N_INPUT, dtype=np.float32)
    vec[cls] = 1.0                                        # one-hot класса (6 чисел)
    vec[6] = np.clip(dt, -5.0, 60.0) / 60.0               # dt, нормированный
    vec[7] = np.clip(width, 0.0, 20.0) / 20.0             # ширина группы
    vec[8] = raw["speed"] / 13.0                          # скорость
    vec[9] = np.clip(gap, 0.0, 60.0) / 60.0               # зазор до второго
    vec[10] = 1.0 if raw["ducking"] else 0.0              # уже присел
    vec[11] = 1.0 if raw["jumping"] else 0.0              # уже в прыжке
    return vec


class QNet(nn.Module):
    """Сеть: 12 входов -> 128 -> 128 -> 3 оценки Q(s,a)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_INPUT, HIDDEN)   # первый слой
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)    # второй слой
        self.fc3 = nn.Linear(HIDDEN, N_ACTIONS) # выход: по числу действий

    def forward(self, x):
        """Прямой проход: батч состояний -> батч троек Q."""
        x = F.relu(self.fc1(x))   # нелинейность, иначе слои схлопнутся в один
        x = F.relu(self.fc2(x))
        return self.fc3(x)        # без активации: Q может быть любым числом


class Agent:
    """Всё обучение: две сети, оптимизатор, буфер, счётчики."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()              # одна сеть на 4 потока
        self.online = QNet().to(DEVICE)           # обучаемая сеть
        self.target = QNet().to(DEVICE)           # замороженная копия для цели
        self.target.load_state_dict(self.online.state_dict())  # старт одинаковый
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=LR)
        self.buffer = deque(maxlen=BUFFER_SIZE)   # старые переходы вытесняются
        self.decisions = 0                        # сколько решений принято
        self.learn_steps = 0                      # сколько шагов обучения
        self.best_distance = 0.0                  # лучшая дистанция за один забег
        self.learning = not PLAY_BEST             # в финальном режиме не учимся
        self.eps_start = EPS_START                # при загрузке best станет меньше
        self.load()                               # продолжить с файла, если есть
    def save(self):
        """Сохранить веса обеих сетей и счётчики."""
        with self.lock:
            data = {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "decisions": self.decisions,
                "learn_steps": self.learn_steps,
                "best_distance": self.best_distance,
                "metric": METRIC_VERSION,
            }
            torch.save(data, self.path)
        log(f"модель сохранена: {self.path}")

    def save_best_from(self, other, distance):
        """Сохранить лучшую модель из проверочной копии, не трогая online."""
        with self.lock:
            self.best_distance = distance
            data = {
                "online": other.online.state_dict(),
                "target": other.target.state_dict(),
                "decisions": self.decisions,
                "learn_steps": self.learn_steps,
                "best_distance": distance,
                "metric": METRIC_VERSION,
            }
            torch.save(data, BEST_FILE)
        log(f"лучшая модель сохранена: дистанция {distance}")

    def save_best(self, distance):
        """Сохранить текущие веса online/target как лучшие."""
        self.save_best_from(self, distance)

    def clone_for_eval(self):
        """Копия текущих весов: проверочный забег без обучения и случайности."""
        clone = Agent.__new__(Agent)      # не вызываем __init__/load
        clone.path = self.path
        clone.lock = threading.Lock()
        clone.online = QNet().to(DEVICE)
        clone.target = QNet().to(DEVICE)
        with self.lock:
            clone.online.load_state_dict(self.online.state_dict())
            clone.target.load_state_dict(self.target.state_dict())
        clone.buffer = deque(maxlen=1)
        clone.decisions = self.decisions
        clone.learn_steps = self.learn_steps
        clone.best_distance = self.best_distance
        clone.learning = False            # push() ничего не делает, epsilon() = 0
        return clone

    def clone_for_behavior(self, epsilon):
        """Замороженная копия весов на один эпизод: действия стабильны,
        а обучение продолжает идти в основной online-сети."""
        clone = self.clone_for_eval()
        clone.learning = False
        clone._behavior_epsilon = epsilon
        return clone

    def load(self):
        """Загрузить модель. Лучшая модель приоритетнее последней."""
        source = BEST_FILE if BEST_FILE.exists() else self.path
        if not source.exists():
            log(f"сеть создана с нуля ({sum(p.numel() for p in self.online.parameters())} параметров)")
            return
        try:
            data = torch.load(source, map_location=DEVICE, weights_only=False)
            self.online.load_state_dict(data["online"])
            self.target.load_state_dict(data["target"])
            # счётчики не откатываем: иначе случайных действий станет больше
            counters = data
            if source != self.path and self.path.exists():
                try:
                    counters = torch.load(self.path, map_location=DEVICE, weights_only=False)
                except Exception:
                    pass
            self.decisions = max(int(data.get("decisions", 0)), int(counters.get("decisions", 0)))
            self.learn_steps = max(int(data.get("learn_steps", 0)), int(counters.get("learn_steps", 0)))
            if data.get("metric") == METRIC_VERSION:
                self.best_distance = float(data.get("best_distance", 0.0))
            else:
                self.best_distance = 0.0   # старый рекорд измерен по другой методике
                log("старый рекорд дистанции сброшен: методика изменилась")
            # после лучшей модели начинаем с малого исследования
            self.eps_start = RESUME_EPS_START if source == BEST_FILE else EPS_START
            log(
                f"модель загружена ({source.name}): решений {self.decisions}, "
                f"шагов обучения {self.learn_steps}, лучшая дистанция {self.best_distance:.0f}, "
                f"epsilon_start {self.eps_start:.2f}"
            )
        except Exception as error:
            log(f"не удалось загрузить модель ({error}), учим с нуля")

    def epsilon(self):
        """Доля случайных действий. В финальном режиме всегда 0."""
        if not self.learning:
            return 0.0
        progress = min(1.0, self.decisions / max(1, EPS_DECAY_DECISIONS))
        return self.eps_start + (EPS_END - self.eps_start) * progress

    def act(self, vec, epsilon, allow_jump=True):
        """Выбрать действие: случайное (исследование) или лучшее по сети.

        allow_jump=False маскирует прыжок: без препятствия рядом он не нужен.
        """
        with self.lock:
            self.decisions += 1
        if random.random() < epsilon:      # исследование
            if allow_jump:
                return random.randrange(N_ACTIONS)
            return random.choice((ACTION_RUN, ACTION_DUCK))
        with self.lock, torch.no_grad():   # без градиентов: это только выбор
            q = self.online(torch.from_numpy(vec).unsqueeze(0).to(DEVICE))
        if not allow_jump:
            q[0, ACTION_JUMP] = -float("inf")  # запрещённое действие
        return int(q.argmax(1).item())     # жадное действие

    def push(self, state, action, reward, next_state, done, discount=1.0):
        """Положить переход в буфер и, если пора, сделать шаг обучения."""
        if not self.learning:      # финальный забег: только игра
            return
        with self.lock:
            self.buffer.append((state, action, reward, next_state, done, discount))
            ready = len(self.buffer) >= 1000 and self.decisions % TRAIN_EVERY == 0
        if ready:
            self.learn()

    def learn(self):
        """Один шаг градиента по случайному батчу из буфера."""
        with self.lock:
            batch = random.sample(self.buffer, BATCH)  # случайность рвёт корреляцию кадров

        states = torch.from_numpy(np.stack([b[0] for b in batch])).to(DEVICE)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=DEVICE)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=DEVICE)
        next_states = torch.from_numpy(np.stack([b[3] for b in batch])).to(DEVICE)
        dones = torch.tensor([float(b[4]) for b in batch], dtype=torch.float32, device=DEVICE)
        discounts = torch.tensor([b[5] for b in batch], dtype=torch.float32, device=DEVICE)

        with self.lock:
            q_values = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                # Double DQN: действие выбирает online, оценивает target
                next_actions = self.online(next_states).argmax(1, keepdim=True)
                next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
                # цель Беллмана; в терминальном состоянии будущего нет
                target = rewards + (1.0 - dones) * discounts * next_q
            loss = F.smooth_l1_loss(q_values, target)  # Huber: устойчив к выбросам
            self.optimizer.zero_grad()                 # обнулить старые градиенты
            loss.backward()                            # посчитать новые
            nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)  # ограничить шаг
            self.optimizer.step()                      # сдвинуть веса
            self.learn_steps += 1
            if self.learn_steps % TARGET_SYNC == 0:    # периодически обновить target
                self.target.load_state_dict(self.online.state_dict())


def next_episode():
    """Выдать номер следующего эпизода или None, если лимит исчерпан."""
    global episodes_started
    with episode_lock:
        if episodes_started >= EPISODES:
            return None
        episodes_started += 1
        return episodes_started


def start_game(page):
    """Начать новый забег: дистанция и препятствия сбрасываются."""
    try:
        page.evaluate(RESTART_JS)
    except Exception:
        pass
    deadline = time.time() + 5.0
    while time.time() < deadline:
        raw = read_game(page)
        if raw["status"] == "OK":
            return True
        if raw["status"] == "HIDDEN":
            return False
        try:
            page.keyboard.press("Space")  # запасной путь, если restart не сработал
        except Exception:
            pass
        time.sleep(0.2)
    return False


def play_episode(page, agent, episode, max_seconds=None, actor=None,
                 epsilon_override=None):
    """Один эпизод: решения, награды, запись переходов в буфер.

    max_seconds переопределяет MAX_EPISODE_SECONDS (для проверки рекорда).
    actor — замороженная политика, которой играет эпизод; agent — сеть,
    которая продолжает учиться на собранных переходах.
    """
    actor = actor or agent
    if not start_game(page):
        return None

    raw = read_game(page)
    if raw["status"] != "OK":
        return None

    limit = MAX_EPISODE_SECONDS if max_seconds is None else max_seconds

    vec = encode(raw)                     # текущее состояние
    passed = int(raw.get("passed", 0))    # сколько препятствий уже позади
    duck_held = False                     # зажата ли стрелка вниз
    launch_vec = None                     # состояние, из которого начат прыжок
    flight_return = 0.0                   # накопленная награда за полёт
    flight_gamma = 1.0                    # дисконт за время полёта
    started = time.time()
    steps = 0
    total_reward = 0.0
    end_reason = "OK"

    while True:
        # --- выходы из эпизода ---
        if stop_event.is_set():
            end_reason = "STOP"
            break
        if MAX_STEPS and steps >= MAX_STEPS:
            end_reason = "LIMIT"
            break
        if limit and time.time() - started > limit:
            end_reason = "TIME"
            break

        airborne = bool(raw["jumping"])
        epsilon = (
            agent.epsilon() if epsilon_override is None else epsilon_override
        )

        # --- выбор и применение действия ---
        if airborne:
            # в воздухе решение уже принято: только отпускаем присед
            if duck_held:
                page.keyboard.up("ArrowDown")
                duck_held = False
            action = ACTION_RUN
            vec = encode(raw)
        else:
            # прыжок разрешён только в зоне препятствия, иначе только бег/присед
            action = actor.act(vec, epsilon, jump_allowed(raw))
            forced = forced_bird_action(raw)
            if forced is not None:
                action = forced
            try:
                if action == ACTION_JUMP:
                    if duck_held:                 # из приседа нельзя прыгнуть сразу
                        page.keyboard.up("ArrowDown")
                        duck_held = False
                    if not raw["jumping"]:
                        page.keyboard.press("Space")   # полный прыжок
                elif action == ACTION_DUCK:
                    if not duck_held:
                        page.keyboard.down("ArrowDown")  # удержание
                        duck_held = True
                else:
                    if duck_held:
                        page.keyboard.up("ArrowDown")    # встать
                        duck_held = False
            except Exception:
                break

        time.sleep(STEP_DELAY)          # дать игре прожить шаг
        next_raw = read_game(page)      # новое состояние
        steps += 1

        # --- падение ---
        if next_raw["status"] == "DEAD":
            end_reason = "DEAD"
            total_reward += CRASH_REWARD
            if airborne and launch_vec is not None:
                # штраф приписываем прыжку, а не состоянию приземления
                final = flight_return + flight_gamma * CRASH_REWARD
                agent.push(launch_vec, ACTION_JUMP, final, launch_vec, True)
            elif not airborne:
                agent.push(vec, action, CRASH_REWARD, vec, True)
            if duck_held:
                try:
                    page.keyboard.up("ArrowDown")
                except Exception:
                    pass
            break

        if next_raw["status"] != "OK":   # пауза или ошибка страницы
            end_reason = next_raw["status"]
            break

        # --- награда за шаг ---
        next_vec = encode(next_raw)
        reward = DIST_REWARD * max(0.0, next_raw["dist"] - raw["dist"])

        next_passed = int(next_raw.get("passed", 0))
        if next_passed < passed:         # препятствие ушло за экран - счётчик уменьшился
            passed = next_passed
        reward += PASS_REWARD * (next_passed - passed)
        passed = next_passed

        total_reward += reward

        # --- запись перехода ---
        if airborne:
            flight_return += flight_gamma * reward   # копим доход полёта
            flight_gamma *= GAMMA
            if not next_raw["jumping"]:              # приземлились
                if launch_vec is not None:
                    agent.push(launch_vec, ACTION_JUMP, flight_return, next_vec, False, flight_gamma)
                launch_vec = None
        elif action == ACTION_JUMP and next_raw["jumping"]:
            launch_vec = vec                         # запомнили точку старта
            flight_return = reward
            flight_gamma = GAMMA
        else:
            agent.push(vec, action, reward, next_vec, False)  # обычный шаг

        raw = next_raw
        vec = next_vec

    if duck_held:
        try:
            page.keyboard.up("ArrowDown")
        except Exception:
            pass

    return {
        "steps": steps,
        "distance": int(raw.get("dist", 0) or 0),
        "reward": total_reward,
        "reason": end_reason,
        "seconds": time.time() - started,
    }


def run_record_attempt(page, agent, stats, window_id, episode):
    """Проверка текущих весов без исследования: может побить рекорд и обновить best."""
    log(f"окно {window_id}: проверка рекорда, эпизод {episode}")
    clone = agent.clone_for_eval()
    result = play_episode(page, clone, episode, max_seconds=RECORD_ATTEMPT_SECONDS)
    if result is None:
        log(f"окно {window_id}: проверка не запустилась")
        return
    if result["reason"] not in ("DEAD", "TIME"):
        log(f"окно {window_id}: проверка отброшена ({result['reason']})")
        return
    if stats.record_eval(result["distance"]):
        agent.save_best_from(clone, result["distance"])
    result["reason"] = "EVAL_" + result["reason"]
    write_result(window_id, episode, result, 0.0, curriculum_stage + 1, stats.best)
    log(
        f"окно {window_id} | проверка {episode:4d} | "
        f"дистанция {result['distance']:6d} | рекорд {stats.best} | {result['reason']}"
    )


def window_worker(window_id, agent, stats):
    """Одно окно Chrome: свой браузер, общая сеть и буфер."""
    position = f"--window-position={(window_id % 2) * 720},{(window_id // 2) * 340}"
    log(f"окно {window_id}: запуск Chrome")
    with sync_playwright() as p:  # у каждого потока свой Playwright
        browser = p.chromium.launch(
            channel=BROWSER_CHANNEL,  # "chrome" или None для Chromium
            headless=HEADLESS,
            args=[
                "--mute-audio",
                position,
                "--window-size=700,320",
                # не тормозить фоновые окна: иначе игра "замерзает"
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-features=CalculateNativeWinOcclusion",
            ],
        )
        page = browser.new_page(viewport={"width": 700, "height": 260})
        page.add_init_script(NO_PAUSE_JS)  # до загрузки игры убрать паузу на blur
        page.goto(GAME_URL, wait_until="domcontentloaded")
        time.sleep(1.5)  # дать загрузиться спрайтам
        # финал: полная скорость (появляются птеродактили);
        # обучение: этап TRAIN_STAGE (MAX_SPEED=13.0)
        max_speed = FINAL_MAX_SPEED if PLAY_BEST else current_max_speed()
        page.evaluate("v => { Runner.config.MAX_SPEED = v; }", max_speed)

        deadline = time.time() + TIME_LIMIT_MINUTES * 60 if TIME_LIMIT_MINUTES else None
        hidden = False

        while not stop_event.is_set():
            if deadline is not None and time.time() > deadline:
                stop_event.set()
                break
            if page.is_closed() or not browser.is_connected():
                log(f"окно {window_id}: браузер закрыт, поток завершён")
                break
            if not page_visible(page):      # свёрнутое окно - не тренируемся
                if not hidden:
                    log(f"окно {window_id}: окно скрыто, результаты не пишутся")
                    hidden = True
                time.sleep(2.0)
                continue
            if hidden:
                log(f"окно {window_id}: окно снова видно")
                hidden = False
            episode = next_episode()
            if episode is None:
                break

            try:
                if agent.learning:
                    eps = agent.epsilon()
                    actor = agent.clone_for_behavior(eps)
                    result = play_episode(
                        page, agent, episode,
                        actor=actor, epsilon_override=eps,
                    )
                else:
                    result = play_episode(page, agent, episode)
            except Exception as error:
                log(f"окно {window_id}: страница закрыта/ошибка: {error}")
                break
            if result is None:
                log(f"окно {window_id}: игра не запустилась")
                continue
            if result["reason"] in ("NOT_PLAYING", "HIDDEN", "ERROR"):
                log(f"окно {window_id}: эпизод {episode} отброшен ({result['reason']})")
                continue

            improved = stats.record(
                result, update_best=TRAIN_GREEDY or PLAY_BEST
            )
            if improved:
                agent.save_best(result["distance"])
            write_result(
                window_id, episode, result,
                agent.epsilon(), curriculum_stage + 1, stats.best,
            )
            if episode % SAVE_EVERY == 0 and agent.learning:
                agent.save()

            log(
                f"окно {window_id} | эпизод {episode:4d} | "
                f"дистанция {result['distance']:6d} | шагов {result['steps']:5d} | "
                f"награда {result['reward']:7.1f} | eps {agent.epsilon():.2f} | "
                f"этап {curriculum_stage + 1} | {result['reason']}"
            )
            if episode % 20 == 0:
                log(stats.summary())
                if maybe_advance_curriculum(stats.recent_average()):
                    page.evaluate("v => { Runner.config.MAX_SPEED = v; }", current_max_speed())

            if agent.learning and not TRAIN_GREEDY and episode % RECORD_EVERY == 0:
                run_record_attempt(page, agent, stats, window_id, episode)

        browser.close()
    log(f"окно {window_id}: завершено")


class Stats:
    """Общая статистика по всем окнам."""

    def __init__(self, best_distance=0.0):
        self.lock = threading.Lock()
        self.episodes = 0
        self.best = best_distance          # лучшая дистанция, из best_dqn.pt
        self.recent = deque(maxlen=50)     # последние 50 дистанций

    def record(self, result, update_best=False):
        """Учесть эпизод. При update_best обновить рекорд и вернуть True."""
        with self.lock:
            self.episodes += 1
            self.recent.append(result["distance"])
            if update_best and result["distance"] > self.best:
                self.best = result["distance"]
                return True
        return False

    def record_eval(self, distance):
        """Учесть проверочный забег: только рекорд, без средней за 50."""
        with self.lock:
            if distance > self.best:
                self.best = distance
                return True
        return False

    def recent_average(self):
        with self.lock:
            return sum(self.recent) / len(self.recent) if self.recent else 0.0

    def summary(self):
        with self.lock:
            average = sum(self.recent) / len(self.recent) if self.recent else 0
            return (
                f"ИТОГО: эпизодов {self.episodes}, средняя за 50: {average:.0f}, "
                f"лучшая дистанция {self.best}"
            )


def main():
    """Точка входа: загрузить модель, поднять окна, ждать, сохранить."""
    global curriculum_stage

    log(f"устройство: {DEVICE}")
    if PLAY_BEST:
        log(
            f"режим: финальный забег по best_dqn.pt, "
            f"MAX_SPEED={FINAL_MAX_SPEED}, без обучения и случайности"
        )
    else:
        curriculum_stage = max(0, min(TRAIN_STAGE, len(CURRICULUM) - 1))
        log(
            f"режим: обучение, этап {curriculum_stage + 1}, "
            f"MAX_SPEED={current_max_speed()}"
        )
    init_results_file()        # results.csv появится сразу
    agent = Agent(MODEL_FILE)  # сеть + буфер, общие для всех окон

    if not PLAY_BEST and TRAIN_STAGE > 0:
        low_best = BEST_FILE.with_name("best_dqn_low.pt")
        if BEST_FILE.exists() and not low_best.exists():
            shutil.copy2(BEST_FILE, low_best)
            log(f"low-speed best сохранён в {low_best.name}")
        agent.best_distance = 0.0
        stats = Stats(0.0)
        log("high-speed best: отсчёт с 0, старый best сохранён")
    else:
        stats = Stats(agent.best_distance)

    threads = []
    for window_id in range(WINDOWS):
        thread = threading.Thread(target=window_worker, args=(window_id, agent, stats), daemon=True)
        thread.start()
        threads.append(thread)
        time.sleep(1.0)  # не стартовать 4 Chrome одновременно

    try:
        while any(thread.is_alive() for thread in threads):
            for thread in threads:
                thread.join(timeout=0.5)  # ждём, периодически проверяя флаг остановки
    except KeyboardInterrupt:
        log("остановка по Ctrl+C")
        stop_event.set()

    stop_event.set()
    for thread in threads:
        thread.join(timeout=10)
    if agent.learning:
        agent.save()  # финальное сохранение
        log("обучение завершено")
    else:
        log("финальный забег завершён")


if __name__ == "__main__":
    main()
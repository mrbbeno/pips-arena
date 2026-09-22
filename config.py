"""All tunable constants in one place. Kept free of torch so the engine imports stay light."""

# --- device: "auto" = CUDA if available, else CPU. Or force "cuda" / "cpu".
DEVICE = "auto"

# --- arena / physics (units = pixels, time = ticks at 60 ticks/s)
FPS = 60
W, H = 1920, 1080          # world size in units; the camera shows part of it (16:9)
REF_W, REF_H = 800, 600    # fixed length scale for the AI's *relative* features (independent of map size)
REF_DIAG = 1000.0
AGENT_R = 14
AGENT_SPEED = 4.0          # max units per tick
AGENT_ACCEL = 0.35         # fraction of (target_vel - vel) applied per tick
MAX_HP = 5
PROJ_R = 4
PROJ_SPEED = 10.0
COOLDOWN = 18              # ticks between shots
MAX_PROJ = 16              # projectile slots per agent
MAX_TICKS = 60 * FPS       # round timeout -> draw
SPAWN_MARGIN = 40
SPAWN_MIN_DIST = 350
SPAWN_MAX_DIST = 1000      # 1v1 spawns: not across the whole map
OBSTACLES = []             # default map for Arena(): list of (x0, y0, x1, y1) rects
MAX_OBST = 24              # obstacle slots per env (unused slots are parked far outside the arena)

# --- weapons: index -> properties. 0 is the default blaster (infinite ammo).
WEAPON_NAMES = ["blaster", "shotgun", "rapid", "rail"]
W_COOLDOWN = [18, 32, 7, 42]
W_SPEED = [10.0, 9.0, 11.0, 20.0]
W_PELLETS = [1, 3, 1, 1]
W_SPREAD_DEG = [0.0, 13.0, 0.0, 0.0]       # angle between pellets
W_DAMAGE = [1, 1, 1, 2]
W_LIFE = [200, 36, 200, 200]               # ticks before a projectile fizzles (shotgun = short range)
W_AMMO = [0, 8, 30, 5]

# --- pickups
ITEM_NAMES = ["shotgun", "rapid", "rail", "shield", "heal"]
ITEM_WEIGHTS = [0.22, 0.22, 0.16, 0.22, 0.18]
MAX_ITEMS = 5              # simultaneously on the map
ITEM_R = 11
ITEM_FIRST = 3 * FPS       # first spawn after round start
ITEM_EVERY = 4 * FPS       # then one every N ticks while below MAX_ITEMS
SHIELD_ADD, SHIELD_MAX = 2, 3
HEAL_ADD = 2

# --- rewards
HIT_REWARD = 1.0
WIN_REWARD = 5.0
TICK_PENALTY = 0.002
DRAW_PENALTY = 3.0         # both agents lose this on a draw (timeout) -> no mutual hiding behind cover
PICKUP_REWARD = 0.2

# --- model
HIDDEN = 256

# --- PPO
N_ENVS = 512               # parallel games; 2 agents each -> 1024 samples per tick
ROLLOUT = 128
EPOCHS = 4
MINIBATCH = 8192
GAMMA = 0.995
LAMBDA = 0.95
CLIP = 0.2
LR = 3e-4
ENT_START = 0.01
ENT_END = 0.001
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5

# --- self-play opponent pool
POOL_FRAC = 0.2            # fraction of envs where agent 1 is a frozen past snapshot
POOL_EVERY = 10            # iterations between snapshots
POOL_SIZE = 20

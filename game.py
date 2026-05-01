import sys, time, json, math, base64, random, hashlib, os, eventlet, threading, requests
from collections import deque

try:
	import numpy as np
except ImportError:
	np = None

# type: 0=empty 1=mountain 2=swamp -1=city -2=general

default_width = 45
duel_width = 34
max_city_ratio = 0.04
max_swamp_ratio = 0.16
max_mountain_ratio = 0.24
left_game = 52
surrender_delay_turns = 100
duel_min_enemy_distance = 15
two_team_min_enemy_distance = 17
default_min_enemy_distance = 9
team_ally_max_distance = 8
spawn_edge_soft_distance = 4


def chkconn(grid_type, n, m):
	fa = [i for i in range(n * m)]
	sz = [1] * (n * m)

	def find(x):
		if x == fa[x]:
			return x
		fa[x] = find(fa[x])
		return fa[x]

	def merge(x, y):
		x = find(x)
		y = find(y)
		if x != y:
			sz[y] += sz[x]
			fa[x] = y
	cnt = 0
	for i in range(n):
		for j in range(m):
			if grid_type[i][j] != 1:
				if i + 1 < n and grid_type[i + 1][j] != 1:
					merge(i * m + j, i * m + j + m)
				if j + 1 < m and grid_type[i][j + 1] != 1:
					merge(i * m + j, i * m + j + 1)
			else:
				cnt += 1
	for i in range(n * m):
		if sz[i] > (n * m - cnt) * 0.9:
			return (i // m, i % m)
	return (-1, -1)


def get_st(grid_type, n, m, res):
	fa = [i for i in range(n * m)]
	sz = [1] * (n * m)

	def find(x):
		if x == fa[x]:
			return x
		fa[x] = find(fa[x])
		return fa[x]

	def merge(x, y):
		x = find(x)
		y = find(y)
		if x != y:
			sz[y] += sz[x]
			fa[x] = y
	cnt = 0
	for i in range(n):
		for j in range(m):
			if grid_type[i][j] != 1:
				if i + 1 < n and grid_type[i + 1][j] != 1:
					merge(i * m + j, i * m + j + m)
				if j + 1 < m and grid_type[i][j + 1] != 1:
					merge(i * m + j, i * m + j + 1)
			else:
				cnt += 1
	max_sz = 0
	for i in range(n * m):
		max_sz = max(max_sz, sz[i])
	for i in range(n * m):
		if sz[find(i)] == max_sz:
			res[i // m][i % m] = True


def get_diff(a, b):
	if np is not None and isinstance(a, np.ndarray):
		a_flat = a.reshape(-1)
		if b is None or len(b) != a_flat.size:
			idx = np.arange(a_flat.size, dtype=np.int64)
		else:
			b_flat = b.reshape(-1) if isinstance(b, np.ndarray) else np.asarray(b, dtype=a_flat.dtype).reshape(-1)
			idx = np.flatnonzero(a_flat != b_flat)
		if idx.size == 0:
			return []
		res = np.empty(idx.size * 2, dtype=np.int64)
		res[0::2] = idx
		res[1::2] = a_flat[idx]
		return res.tolist()
	res = []
	b_len = -1 if b is None else len(b)
	for i in range(len(a)):
		if b_len != len(a) or a[i] != b[i]:
			res.append(i)
			res.append(a[i])
	return res


class Game:
	def __init__(self, game_conf, update, emit_init_map, player_ids, rplayer_ids, chat_message, gid, md5, end_game):
		print('start game:', gid, player_ids, game_conf['player_names'])
		sys.stdout.flush()
		self.otime = time.time()
		self.md5 = md5
		self.end_game = end_game
		self.player_ids = player_ids
		self.player_ids_rev = {}
		for i in range(len(player_ids)):
			self.player_ids_rev[player_ids[i]] = i
		self.update = update
		self.pcnt = len(player_ids)
		self.speed = game_conf['speed']
		self.names = game_conf['player_names']
		self.team = game_conf['player_teams']
		self.colors = game_conf.get('player_colors', [i + 1 for i in range(len(player_ids))])
		self.rpcnt = 0
		for i in self.team:
			if i:
				self.rpcnt += 1
		self.active_players = [i for i in range(len(player_ids)) if self.team[i]]
		self.active_teams = sorted(set([self.team[i] for i in self.active_players]))
		self.is_duel = len(self.active_players) == 2 and len(self.active_teams) == 2
		self.is_two_team_game = len(self.active_teams) == 2
		self.team_players = {}
		for i in self.active_players:
			self.team_players.setdefault(self.team[i], []).append(i)
		self.width_ratio = game_conf['width_ratio'] / 2 + 0.5
		self.height_ratio = game_conf['height_ratio'] / 2 + 0.5
		self.city_ratio = game_conf['city_ratio']
		self.mountain_ratio = game_conf['mountain_ratio']
		self.swamp_ratio = game_conf['swamp_ratio']
		self.spawn_fairness = game_conf.get('spawn_fairness', 0.75)
		self.city_fairness = game_conf.get('city_fairness', 0.75)
		self.move_general_on_capture = game_conf.get('move_general_on_capture', False)
		self.city_state = game_conf.get('city_state', False) and game_conf.get('custom_map', '').strip() == ''
		self.pstat = [0 for i in player_ids]
		self.pmove = [[] for i in player_ids]
		self.lst_move = [(-1, -1, -1, -1, False) for i in player_ids]
		self.watching = [True for i in player_ids]
		self.spec = [False for i in player_ids]
		self.grid_type_lst = [[] for i in player_ids]
		self.army_cnt_lst = [[] for i in player_ids]
		self.deadorder = [0] * len(player_ids)
		self.deadcount = 0
		self.surrender_turn = [0] * len(player_ids)
		self.chat_message = chat_message
		self.gid = gid
		self.lock = threading.RLock()
		self.turn = 0
		self.map_generation_announced = False
		self.recentkills = {}
		self.history = []
		self.history_grid_type_lst = []
		self.history_army_cnt_lst = []
		if game_conf['custom_map'] != '':
			self.getcustommap(game_conf['custom_map'])
		else:
			self.genmap()
		self.sel_generals()
		if not self.is_custom:
			if self.city_state:
				self.place_city_states()
			self.place_neutral_cities()
		self.enable_fast_arrays()
		for i in range(self.pcnt):
			emit_init_map(self.player_ids[i], {'n': self.n, 'm': self.m, 'player_ids': rplayer_ids, 'player_colors': self.colors, 'general': self.generals[i]})

	def enable_fast_arrays(self):
		if np is None:
			return
		self.grid_type = np.asarray(self.grid_type, dtype=np.int16)
		self.owner = np.asarray(self.owner, dtype=np.int16)
		self.army_cnt = np.asarray(self.army_cnt, dtype=np.int64)
		self.st = np.asarray(self.st, dtype=np.bool_)

	def numpy_fast_path(self):
		return np is not None and isinstance(self.owner, np.ndarray)

	def send_message(self, sid, data):
		id = self.player_ids_rev[sid]
		uid = self.names[id]
		if data['team']:
			for i in range(self.pcnt):
				if self.team[i] == self.team[id]:
					self.chat_message(self.player_ids[i], 'sid', uid, self.colors[id], data['text'], True)
		else:
			self.chat_message(self.gid, 'room', uid, self.colors[id], data['text'])

	def send_system_message(self, text):
		self.chat_message(self.gid, 'room', '', 0, text)

	def getcustommap(self, title):
		try:
			r = requests.get('http://generals.io/api/map', params={'name': title.encode('utf-8')}).json()
			n = r['height']
			m = r['width']
			t = r['map'].split(',')
			self.owner = [[0 for j in range(m)] for i in range(n)]
			self.army_cnt = [[0 for j in range(m)] for i in range(n)]
			self.grid_type = [[0 for j in range(m)] for i in range(n)]
			for i in range(n):
				for j in range(m):
					x = t[i * m + j].strip(' ')
					if x == 'm':
						self.grid_type[i][j] = 1
					elif x == 's':
						self.grid_type[i][j] = 2
					elif x == 'g':
						self.grid_type[i][j] = -2
					elif x == 's':
						self.grid_type[i][j] = 2
					elif len(x):
						if x[0] == 'n':
							self.army_cnt[i][j] = int(x[1:])
						else:
							self.army_cnt[i][j] = int(x)
							self.grid_type[i][j] = -1
			self.n = n
			self.m = m
			self.st = [[False for j in range(m)] for i in range(n)]
			get_st(self.grid_type, n, m, self.st)
			self.is_custom = True
		except:
			self.genmap()

	def base_map_width(self):
		if self.is_duel:
			return duel_width
		return default_width + max(0, self.rpcnt - 8) * 2

	def genmap(self):
		self.is_custom = False
		base_width = self.base_map_width()
		ni = random.randint(base_width - 5, base_width + 5)
		mi = base_width * base_width // ni
		self.n = n = int(ni * self.height_ratio)
		self.m = m = int(mi * self.width_ratio)
		swamp_ratio = max_swamp_ratio * self.swamp_ratio
		mountain_ratio = swamp_ratio + max_mountain_ratio * self.mountain_ratio
		while True:
			grid_type = [[0 for j in range(m)] for i in range(n)]
			for i in range(n):
				for j in range(m):
					tmp = random.random()
					grid_type[i][j] = 2 if tmp < swamp_ratio else 1 if tmp < mountain_ratio else 0
			x, y = chkconn(grid_type, n, m)
			if x != -1:
				break
		self.grid_type = grid_type
		self.st = [[False for j in range(m)] for i in range(n)]
		get_st(grid_type, n, m, self.st)
		self.owner = [[0 for j in range(m)] for i in range(n)]
		self.army_cnt = [[0 for j in range(m)] for i in range(n)]

	def make_general_candidate(self):
		ge = []
		sp = []
		for i in range(self.n):
			for j in range(self.m):
				if self.st[i][j]:
					if self.grid_type[i][j] == -2:
						ge.append((i, j))
					elif self.grid_type[i][j] == 0:
						sp.append((i, j))
		random.shuffle(sp)
		if self.rpcnt > len(ge):
			ge.extend(sp[:self.rpcnt - len(ge)])
		while self.rpcnt > len(ge):
			ge.append((-1, -1))
		random.shuffle(ge)
		return ge[:self.rpcnt]

	def spawn_cells(self):
		res = []
		for i in range(self.n):
			for j in range(self.m):
				if self.st[i][j] and self.grid_type[i][j] == 0:
					res.append((i, j))
		return res

	def make_team_general_candidate(self):
		ally_limit = self.ally_spawn_distance_limit()
		if ally_limit is None:
			return self.make_general_candidate()
		sp = self.spawn_cells()
		random.shuffle(sp)
		used = set()
		pos_by_player = {}
		for team in self.active_teams:
			players = self.team_players.get(team, [])
			if not players:
				continue
			anchors = [pos for pos in sp if pos not in used]
			random.shuffle(anchors)
			selected = None
			for anchor in anchors[:80]:
				nearby = [pos for pos in sp if pos not in used and self.manhattan(pos, anchor) <= ally_limit]
				if len(nearby) < len(players):
					continue
				random.shuffle(nearby)
				selected = nearby[:len(players)]
				break
			if selected is None:
				return self.make_general_candidate()
			for i in range(len(players)):
				pos_by_player[players[i]] = selected[i]
				used.add(selected[i])
		return [pos_by_player.get(player, (-1, -1)) for player in self.active_players]

	def active_general_players(self, ge):
		return [(self.active_players[i], ge[i]) for i in range(min(len(self.active_players), len(ge)))]

	def enemy_min_distance(self):
		if self.is_duel:
			return duel_min_enemy_distance
		if self.is_two_team_game and self.rpcnt == 4:
			return two_team_min_enemy_distance
		return default_min_enemy_distance

	def general_min_distance(self):
		return self.enemy_min_distance()

	def ally_spawn_distance_limit(self):
		if any(len(self.team_players.get(team, [])) > 1 for team in self.active_teams):
			return team_ally_max_distance
		return None

	def manhattan(self, a, b):
		return abs(a[0] - b[0]) + abs(a[1] - b[1])

	def edge_distance(self, pos):
		return min(pos[0], pos[1], self.n - 1 - pos[0], self.m - 1 - pos[1])

	def spawn_edge_target_distance(self):
		return min(spawn_edge_soft_distance, max(1, min(self.n, self.m) // 8))

	def spawn_edge_weight(self, pos):
		target = self.spawn_edge_target_distance()
		if target <= 0:
			return 1
		edge = min(self.edge_distance(pos), target)
		return 0.45 + 0.55 * edge / target

	def general_edge_weight(self, ge):
		if self.is_custom:
			return 1
		active = self.active_general_players(ge)
		if not active:
			return 1
		weight = 1
		for _, pos in active:
			if pos != (-1, -1):
				weight *= self.spawn_edge_weight(pos)
		return max(1e-12, weight)

	def valid_team_spawn_distances(self, active):
		ally_limit = self.ally_spawn_distance_limit()
		if ally_limit is None:
			return True
		by_team = {}
		for player, pos in active:
			by_team.setdefault(self.team[player], []).append(pos)
		for positions in by_team.values():
			if len(positions) <= 1:
				continue
			if len(positions) == 2:
				if self.manhattan(positions[0], positions[1]) > ally_limit:
					return False
				continue
			for i in range(len(positions)):
				if min(self.manhattan(positions[i], positions[j]) for j in range(len(positions)) if i != j) > ally_limit:
					return False
			max_team_distance = max(self.manhattan(positions[i], positions[j]) for i in range(len(positions)) for j in range(i))
			if max_team_distance > ally_limit * 2:
				return False
		return True

	def choose_city_state_slot(self, general, reserved=None):
		if reserved is None:
			reserved = set()
		if general == (-1, -1):
			return None
		x, y = general
		candidates = []
		for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
			nx = x + dx
			ny = y + dy
			if self.chkxy(nx, ny) and self.grid_type[nx][ny] == 0 and self.owner[nx][ny] == 0 and (nx, ny) not in reserved:
				candidates.append((nx, ny))
		random.shuffle(candidates)
		return candidates[0] if candidates else None

	def valid_general_candidate(self, ge):
		active = self.active_general_players(ge)
		if len(active) < self.rpcnt:
			return False
		for _, pos in active:
			if pos == (-1, -1):
				return False
		for i in range(len(active)):
			player_i, pos_i = active[i]
			for j in range(i):
				player_j, pos_j = active[j]
				tdis = self.manhattan(pos_i, pos_j)
				if tdis <= 1:
					return False
				if self.team[player_i] != self.team[player_j] and tdis < self.enemy_min_distance():
					return False
		if not self.valid_team_spawn_distances(active):
			return False
		if self.city_state:
			reserved = set()
			for _, pos in active:
				slot = self.choose_city_state_slot(pos, reserved)
				if slot is None:
					return False
				reserved.add(slot)
		return True

	def distance_map(self, start, max_dist=None):
		dist = [[-1 for j in range(self.m)] for i in range(self.n)]
		if start == (-1, -1) or not self.chkxy(start[0], start[1]) or self.grid_type[start[0]][start[1]] == 1:
			return dist
		q = deque([start])
		dist[start[0]][start[1]] = 0
		while q:
			x, y = q.popleft()
			if max_dist is not None and dist[x][y] >= max_dist:
				continue
			for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
				nx = x + dx
				ny = y + dy
				if self.chkxy(nx, ny) and dist[nx][ny] == -1 and self.grid_type[nx][ny] != 1:
					dist[nx][ny] = dist[x][y] + 1
					q.append((nx, ny))
		return dist

	def distance_map_many(self, starts, max_dist=None):
		dist = [[-1 for j in range(self.m)] for i in range(self.n)]
		q = deque()
		for start in starts:
			if start == (-1, -1) or not self.chkxy(start[0], start[1]) or self.grid_type[start[0]][start[1]] == 1:
				continue
			dist[start[0]][start[1]] = 0
			q.append(start)
		while q:
			x, y = q.popleft()
			if max_dist is not None and dist[x][y] >= max_dist:
				continue
			for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
				nx = x + dx
				ny = y + dy
				if self.chkxy(nx, ny) and dist[nx][ny] == -1 and self.grid_type[nx][ny] != 1:
					dist[nx][ny] = dist[x][y] + 1
					q.append((nx, ny))
		return dist

	def spawn_profile(self, general):
		if not hasattr(self, 'spawn_profile_cache'):
			self.spawn_profile_cache = {}
		if general in self.spawn_profile_cache:
			return self.spawn_profile_cache[general]
		dist = self.distance_map(general, 12)
		open_sides = 0
		for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
			nx = general[0] + dx
			ny = general[1] + dy
			if self.chkxy(nx, ny) and self.grid_type[nx][ny] != 1:
				open_sides += 1
		land_r4 = 0
		land_r8 = 0
		land_r12 = 0
		choke_count = 0
		for i in range(self.n):
			for j in range(self.m):
				d = dist[i][j]
				if d <= 0:
					continue
				if d <= 4:
					land_r4 += 1
				if d <= 8:
					land_r8 += 1
					degree = 0
					for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
						nx = i + dx
						ny = j + dy
						if self.chkxy(nx, ny) and self.grid_type[nx][ny] != 1:
							degree += 1
					if degree <= 2:
						choke_count += 1
				if d <= 12:
					land_r12 += 1
		first_round_cap = min(25, 1 + land_r4 + open_sides * 3)
		res = {
			'open_sides': open_sides,
			'land_r4': land_r4,
			'land_r8': land_r8,
			'land_r12': land_r12,
			'first_round_cap': first_round_cap,
			'choke_count': choke_count,
		}
		self.spawn_profile_cache[general] = res
		return res

	def rel_diff(self, a, b):
		return abs(a - b) / max(1.0, (a + b) / 2)

	def spawn_unfairness(self, a, b):
		pa = self.spawn_profile(a)
		pb = self.spawn_profile(b)
		return (
			2.0 * abs(pa['open_sides'] - pb['open_sides'])
			+ 1.5 * self.rel_diff(pa['land_r4'], pb['land_r4'])
			+ 1.2 * self.rel_diff(pa['land_r8'], pb['land_r8'])
			+ 0.8 * self.rel_diff(pa['land_r12'], pb['land_r12'])
			+ 1.5 * self.rel_diff(pa['first_round_cap'], pb['first_round_cap'])
			+ 0.3 * self.rel_diff(pa['choke_count'], pb['choke_count'])
		)

	def team_spawn_profile(self, positions):
		profiles = [self.spawn_profile(pos) for pos in positions if pos != (-1, -1)]
		if not profiles:
			return None
		res = {}
		for key in profiles[0]:
			res[key] = sum(profile[key] for profile in profiles) / len(profiles)
		res['weakest_first_round_cap'] = min(profile['first_round_cap'] for profile in profiles)
		return res

	def team_spawn_unfairness(self, ge):
		active = self.active_general_players(ge)
		by_team = {}
		for player, pos in active:
			by_team.setdefault(self.team[player], []).append(pos)
		if len(by_team) != 2:
			return 0
		profiles = [self.team_spawn_profile(by_team[team]) for team in self.active_teams]
		if profiles[0] is None or profiles[1] is None:
			return 0
		pa, pb = profiles
		return (
			1.5 * abs(pa['open_sides'] - pb['open_sides'])
			+ 1.3 * self.rel_diff(pa['land_r4'], pb['land_r4'])
			+ 1.0 * self.rel_diff(pa['land_r8'], pb['land_r8'])
			+ 0.7 * self.rel_diff(pa['land_r12'], pb['land_r12'])
			+ 1.2 * self.rel_diff(pa['first_round_cap'], pb['first_round_cap'])
			+ 1.0 * self.rel_diff(pa['weakest_first_round_cap'], pb['weakest_first_round_cap'])
			+ 0.25 * self.rel_diff(pa['choke_count'], pb['choke_count'])
		)

	def two_team_layout_score(self, ge):
		active = self.active_general_players(ge)
		enemy_distances = []
		ally_distances = []
		for i in range(len(active)):
			player_i, pos_i = active[i]
			for j in range(i):
				player_j, pos_j = active[j]
				tdis = self.manhattan(pos_i, pos_j)
				if self.team[player_i] == self.team[player_j]:
					ally_distances.append(tdis)
				else:
					enemy_distances.append(tdis)
		if not enemy_distances:
			return 1
		enemy_distance = min(enemy_distances)
		enemy_weight = math.exp(-0.06 * max(0, enemy_distance - self.enemy_min_distance()))
		if ally_distances:
			ally_penalty = sum(abs(tdis - min(6, team_ally_max_distance)) / max(1, team_ally_max_distance) for tdis in ally_distances)
			ally_weight = math.exp(-0.25 * ally_penalty)
		else:
			ally_weight = 1
		return max(1e-12, enemy_weight * ally_weight)

	def score_general_candidate(self, ge):
		if len(ge) < self.rpcnt:
			return 0
		for pos in ge[:self.rpcnt]:
			if pos == (-1, -1):
				return 0
		if self.is_duel and not self.is_custom and self.rpcnt == 2:
			tdis = self.manhattan(ge[0], ge[1])
			distance_weight = math.exp(-0.08 * max(0, tdis - self.enemy_min_distance()))
			unfair = self.spawn_unfairness(ge[0], ge[1])
			return max(1e-12, distance_weight * self.general_edge_weight(ge) * math.exp(-0.65 * self.spawn_fairness * unfair))
		if self.is_two_team_game and not self.is_custom:
			unfair = self.team_spawn_unfairness(ge)
			return max(1e-12, self.two_team_layout_score(ge) * self.general_edge_weight(ge) * math.exp(-0.45 * self.spawn_fairness * unfair))
		tv = 0
		for i in range(self.rpcnt):
			for j in range(i):
				tdis = self.manhattan(ge[i], ge[j])
				tv += 0.88**tdis + max(0, 9 - tdis)
		tv = 1 / (tv + 1e-8)
		if self.is_custom:
			return tv**1.2
		return tv**2.2 * self.general_edge_weight(ge)

	def choose_general_candidate(self):
		ges = []
		gevals = []
		attempts = 0
		target_candidates = 500 if self.is_duel else 100 if self.is_two_team_game and self.rpcnt <= 4 else 50 if self.is_two_team_game else 120 if self.rpcnt <= 8 else 20
		max_attempts = target_candidates * (300 if self.rpcnt <= 8 else 200)
		self.spawn_profile_cache = {}
		while len(ges) < target_candidates and attempts < max_attempts:
			attempts += 1
			ge = self.make_team_general_candidate() if not self.is_custom and self.ally_spawn_distance_limit() is not None else self.make_general_candidate()
			if not self.is_custom and not self.valid_general_candidate(ge):
				continue
			tv = self.score_general_candidate(ge)
			if tv <= 0:
				continue
			ges.append(ge)
			gevals.append(tv)
		if not ges:
			for _ in range(target_candidates * 20):
				ge = self.make_general_candidate()
				if not self.is_custom and not self.valid_general_candidate(ge):
					continue
				tv = self.score_general_candidate(ge)
				if tv > 0:
					ges.append(ge)
					gevals.append(tv)
		if not ges:
			for _ in range(target_candidates):
				ge = self.make_general_candidate()
				tv = self.score_general_candidate(ge)
				if tv > 0:
					ges.append(ge)
					gevals.append(tv)
		if not ges:
			return [(-1, -1) for _ in range(self.rpcnt)]
		total = sum(gevals)
		gpos = random.random() * total
		for i in range(len(ges)):
			if gevals[i] > gpos:
				return ges[i]
			gpos -= gevals[i]
		return ges[-1]

	def sel_generals(self):
		ge = self.choose_general_candidate()
		for i in range(self.n):
			for j in range(self.m):
				if self.st[i][j]:
					if self.grid_type[i][j] == -2:
						self.grid_type[i][j] = 0
		self.generals = [(-1, -1)for _ in range(self.pcnt)]
		cu = 0
		for i in range(self.pcnt):
			if self.team[i] == 0:
				self.pstat[i] = left_game
			else:
				if ge[cu] == (-1, -1):
					self.pstat[i] = left_game
				else:
					self.generals[i] = ge[cu]
					self.grid_type[ge[cu][0]][ge[cu][1]] = -2
					self.owner[ge[cu][0]][ge[cu][1]] = i + 1
					self.army_cnt[ge[cu][0]][ge[cu][1]] = 1
				cu += 1

	def place_city(self, pos, army, owner=0):
		x, y = pos
		self.grid_type[x][y] = -1
		self.owner[x][y] = owner
		self.army_cnt[x][y] = army

	def place_city_states(self):
		reserved = set()
		for i in self.active_players:
			if self.generals[i] == (-1, -1):
				continue
			slot = self.choose_city_state_slot(self.generals[i], reserved)
			if slot is None:
				continue
			reserved.add(slot)
			self.place_city(slot, 1)

	def neutral_city_candidates(self):
		res = []
		for i in range(self.n):
			for j in range(self.m):
				if self.st[i][j] and self.grid_type[i][j] == 0 and self.owner[i][j] == 0:
					res.append((i, j))
		return res

	def target_neutral_city_count(self):
		return int(round(self.n * self.m * max_city_ratio * self.city_ratio))

	def place_random_neutral_cities(self, count):
		candidates = self.neutral_city_candidates()
		random.shuffle(candidates)
		placed = 0
		for pos in candidates[:count]:
			self.place_city(pos, random.randint(40, 50))
			placed += 1
		return placed

	def team_general_positions(self, team):
		return [self.generals[i] for i in self.team_players.get(team, []) if self.generals[i] != (-1, -1)]

	def fair_city_choices(self, candidates, own_dist, other_dist, desired_distance, tolerance, center=None):
		choices = []
		for x, y in candidates:
			d = own_dist[x][y]
			if d < 2 or abs(d - desired_distance) > tolerance:
				continue
			od = other_dist[x][y]
			if od != -1 and od + 2 < d:
				continue
			if center is not None and self.manhattan((x, y), center) > 3:
				continue
			choices.append((x, y))
		return choices

	def choose_fair_city_pair(self, dist0, dist1, desired_distance, tolerance, center0=None, center1=None):
		candidates = self.neutral_city_candidates()
		if len(candidates) < 2:
			return None
		a_choices = self.fair_city_choices(candidates, dist0, dist1, desired_distance, tolerance, center0)
		b_choices = self.fair_city_choices(candidates, dist1, dist0, desired_distance, tolerance, center1)
		if not a_choices or not b_choices:
			return None
		a = random.choice(a_choices)
		b_choices = [pos for pos in b_choices if pos != a]
		if not b_choices:
			return None
		return a, random.choice(b_choices)

	def place_fair_two_team_cities(self, target_count):
		if len(self.active_teams) != 2 or target_count < 2:
			return 0
		pair_count = min(target_count // 2, int(round(target_count * self.city_fairness / 2)))
		if pair_count <= 0:
			return 0
		team0, team1 = self.active_teams
		g0 = self.team_general_positions(team0)
		g1 = self.team_general_positions(team1)
		if not g0 or not g1:
			return 0
		dist0 = self.distance_map_many(g0)
		dist1 = self.distance_map_many(g1)
		tolerance = max(1, int(math.ceil(4 - 3 * self.city_fairness)))
		max_city_distance = max(8, min(max(self.n, self.m), 18))
		placed = 0
		placed_pairs = 0
		attempts = 0
		cluster_left = 0
		cluster0 = None
		cluster1 = None
		cluster_distance = None
		while placed_pairs < pair_count and attempts < pair_count * 140:
			attempts += 1
			desired_distance = cluster_distance if cluster_left > 0 else random.randint(2, max_city_distance)
			pair = self.choose_fair_city_pair(
				dist0,
				dist1,
				desired_distance,
				tolerance,
				cluster0 if cluster_left > 0 else None,
				cluster1 if cluster_left > 0 else None,
			)
			if pair is None and cluster_left > 0:
				cluster_left = 0
				cluster0 = None
				cluster1 = None
				cluster_distance = None
				continue
			if pair is None:
				continue
			a, b = pair
			army = random.randint(40, 50)
			self.place_city(a, army)
			self.place_city(b, army)
			placed += 2
			placed_pairs += 1
			if cluster_left > 0:
				cluster_left -= 1
				cluster0 = a
				cluster1 = b
			elif placed_pairs < pair_count and random.random() < 0.65 * self.city_fairness:
				cluster_left = random.randint(1, min(2, pair_count - placed_pairs))
				cluster0 = a
				cluster1 = b
				cluster_distance = desired_distance
			if cluster_left == 0:
				cluster0 = None
				cluster1 = None
				cluster_distance = None
		return placed

	def place_fair_duel_cities(self, target_count):
		return self.place_fair_two_team_cities(target_count)

	def place_neutral_cities(self):
		target_count = min(self.target_neutral_city_count(), len(self.neutral_city_candidates()))
		if target_count <= 0:
			return
		placed = 0
		if self.is_two_team_game and self.city_fairness > 0:
			placed = self.place_fair_two_team_cities(target_count)
		remaining = min(target_count - placed, len(self.neutral_city_candidates()))
		if remaining > 0:
			self.place_random_neutral_cities(remaining)

	def chkxy(self, x, y):
		return x >= 0 and y >= 0 and x < self.n and y < self.m

	def tile_counts(self):
		if self.numpy_fast_path():
			return {
				'mountains': int(np.count_nonzero(self.grid_type == 1)),
				'swamps': int(np.count_nonzero(self.grid_type == 2)),
				'neutral_cities': int(np.count_nonzero((self.grid_type == -1) & (self.owner == 0))),
				'owned_cities': int(np.count_nonzero((self.grid_type == -1) & (self.owner > 0))),
				'generals': int(np.count_nonzero(self.grid_type == -2)),
			}
		res = {
			'mountains': 0,
			'swamps': 0,
			'neutral_cities': 0,
			'owned_cities': 0,
			'generals': 0,
		}
		for i in range(self.n):
			for j in range(self.m):
				if self.grid_type[i][j] == 1:
					res['mountains'] += 1
				elif self.grid_type[i][j] == 2:
					res['swamps'] += 1
				elif self.grid_type[i][j] == -1:
					if self.owner[i][j] > 0:
						res['owned_cities'] += 1
					else:
						res['neutral_cities'] += 1
				elif self.grid_type[i][j] == -2:
					res['generals'] += 1
		return res

	def map_profile_name(self):
		if self.is_custom:
			return 'custom'
		if self.is_duel:
			return '1v1'
		if self.is_two_team_game and self.rpcnt == 4:
			return '2v2'
		if self.is_two_team_game:
			return '2-team'
		return 'FFA'

	def fmt_ratio(self, value):
		return ('%.2f' % value).rstrip('0').rstrip('.')

	def map_generation_message(self):
		counts = self.tile_counts()
		if self.is_custom:
			return (
				'Map loaded: custom, size=%dx%d, players=%d, teams=%d, cities=%d neutral/%d owned, mountains=%d, swamps=%d.'
				% (self.n, self.m, self.rpcnt, len(self.active_teams), counts['neutral_cities'], counts['owned_cities'], counts['mountains'], counts['swamps'])
			)
		ally_limit = self.ally_spawn_distance_limit()
		ally_part = ', allyMax=%d' % ally_limit if ally_limit is not None else ''
		return (
			'Map generated: profile=%s, size=%dx%d, players=%d, teams=%d, density city/mountain/swamp=%s/%s/%s, '
			'spawnFair=%s, cityFair=%s, enemyMin=%d, edgeSoft=%d%s, cities=%d neutral/%d city-state, mountains=%d, swamps=%d.'
			% (
				self.map_profile_name(),
				self.n,
				self.m,
				self.rpcnt,
				len(self.active_teams),
				self.fmt_ratio(self.city_ratio),
				self.fmt_ratio(self.mountain_ratio),
				self.fmt_ratio(self.swamp_ratio),
				self.fmt_ratio(self.spawn_fairness),
				self.fmt_ratio(self.city_fairness),
				self.enemy_min_distance(),
				self.spawn_edge_target_distance(),
				ally_part,
				counts['neutral_cities'],
				counts['owned_cities'],
				counts['mountains'],
				counts['swamps'],
			)
		)

	def announce_map_generation(self):
		if self.map_generation_announced:
			return
		self.map_generation_announced = True
		self.send_system_message(self.map_generation_message())

	def build_leaderboard(self, owner_arr=None, army_arr=None):
		if owner_arr is not None and army_arr is not None:
			owner_flat = owner_arr.reshape(-1)
			army_flat = army_arr.reshape(-1)
			army_by_owner = np.zeros(self.pcnt + 1, dtype=np.int64)
			np.add.at(army_by_owner, owner_flat, army_flat)
			land_by_owner = np.bincount(owner_flat, minlength=self.pcnt + 1)
			pl_v = [[int(army_by_owner[i + 1]), int(land_by_owner[i + 1])] for i in range(self.pcnt)]
		else:
			pl_v = [[0, 0] for i in range(self.pcnt)]
			for i in range(self.n):
				for j in range(self.m):
					if self.owner[i][j]:
						pl_v[self.owner[i][j] - 1][0] += self.army_cnt[i][j]
						pl_v[self.owner[i][j] - 1][1] += 1
		leaderboard = []
		for i in range(self.pcnt):
			cl = ''
			if self.pstat[i] == left_game:
				cl = 'dead'
			elif self.pstat[i]:
				cl = 'afk'
			if self.team[i]:
				leaderboard.append({'team': self.team[i], 'uid': self.names[i], 'army': pl_v[i][0], 'land': pl_v[i][1], 'class_': cl, 'dead': self.deadorder[i], 'id': i + 1, 'color': self.colors[i]})
		return leaderboard

	def team_visible_mask(self, owner_arr, p, force_full=False):
		if force_full or p == -1 or self.team[p] == 0 or self.spec[p]:
			return np.ones(owner_arr.shape, dtype=np.bool_)
		team_lookup = np.asarray([0] + self.team, dtype=np.int16)
		owned = (owner_arr != 0) & (team_lookup[owner_arr] == self.team[p])
		visible = owned.copy()
		visible[1:, :] |= owned[:-1, :]
		visible[:-1, :] |= owned[1:, :]
		visible[:, 1:] |= owned[:, :-1]
		visible[:, :-1] |= owned[:, 1:]
		visible[1:, 1:] |= owned[:-1, :-1]
		visible[1:, :-1] |= owned[:-1, 1:]
		visible[:-1, 1:] |= owned[1:, :-1]
		visible[:-1, :-1] |= owned[1:, 1:]
		return visible

	def encoded_numpy_frame(self, p, grid_arr, owner_arr, army_arr, force_full=False):
		visible = self.team_visible_mask(owner_arr, p, force_full)
		rt = np.where(visible, 200, 202).astype(np.int16)
		rc = np.where(visible, army_arr, 0).astype(np.int64)

		swamp = grid_arr == 2
		rt[swamp & ~visible] = 205
		rt[swamp & visible & (owner_arr == 0)] = 204
		swamp_owned = swamp & visible & (owner_arr > 0)
		rt[swamp_owned] = owner_arr[swamp_owned] + 150

		mountain = grid_arr == 1
		rt[mountain] = np.where(visible[mountain], 201, 203)

		city = grid_arr == -1
		rt[city & ~visible] = 203
		city_visible = city & visible
		rt[city_visible] = owner_arr[city_visible] + 50

		general = grid_arr == -2
		rt[general & ~visible] = 202
		general_visible = general & visible
		rt[general_visible] = owner_arr[general_visible] + 100

		empty_visible = (grid_arr == 0) & visible
		rt[empty_visible] = 200
		empty_occupied = empty_visible & ((owner_arr != 0) | (army_arr != 0))
		rt[empty_occupied] = owner_arr[empty_occupied]
		return rt.reshape(-1), rc.reshape(-1)

	def update_payload(self, rt_flat, rc_flat, tmp, leaderboard, kls, stat, force_full, prev_grid_type, prev_army_cnt):
		if force_full:
			return {
				'grid_type': rt_flat.tolist() if isinstance(rt_flat, np.ndarray) else rt_flat,
				'army_cnt': rc_flat.tolist() if isinstance(rc_flat, np.ndarray) else rc_flat,
				'lst_move': {'x': tmp[0], 'y': tmp[1], 'dx': tmp[2], 'dy': tmp[3], 'half': tmp[4]},
				'leaderboard': leaderboard,
				'turn': self.turn,
				'kills': kls,
				'game_end': stat,
				'is_diff': False,
			}
		return {
			'grid_type': get_diff(rt_flat, prev_grid_type),
			'army_cnt': get_diff(rc_flat, prev_army_cnt),
			'lst_move': {'x': tmp[0], 'y': tmp[1], 'dx': tmp[2], 'dy': tmp[3], 'half': tmp[4]},
			'leaderboard': leaderboard,
			'turn': self.turn,
			'kills': kls,
			'game_end': stat,
			'is_diff': True,
		}

	def sendmap_numpy(self, stat):
		history_hash = None
		grid_arr = self.grid_type
		owner_arr = self.owner
		army_arr = self.army_cnt
		kls = self.recentkills
		self.recentkills = {}
		leaderboard = self.build_leaderboard(owner_arr, army_arr)
		for p in range(-1, self.pcnt):
			if p != -1 and not self.watching[p]:
				continue
			rt_flat, rc_flat = self.encoded_numpy_frame(p, grid_arr, owner_arr, army_arr, stat)
			tmp = self.lst_move[p]
			if p == -1:
				force_full = not self.history or self.turn % 50 == 0 or stat
				prev_grid_type = self.history_grid_type_lst
				prev_army_cnt = self.history_army_cnt_lst
			else:
				force_full = stat or self.turn % 50 == 0 or random.randint(0, 50) == 0
				prev_grid_type = self.grid_type_lst[p]
				prev_army_cnt = self.army_cnt_lst[p]
			res_data = self.update_payload(rt_flat, rc_flat, tmp, leaderboard, kls, stat, force_full, prev_grid_type, prev_army_cnt)
			if history_hash:
				res_data['replay'] = history_hash
			if p != -1:
				self.grid_type_lst[p] = rt_flat.copy()
				self.army_cnt_lst[p] = rc_flat.copy()
				self.lst_move[p] = (-1, -1, -1, -1, False)
				self.update(self.player_ids[p], res_data)
			else:
				self.history_grid_type_lst = rt_flat.copy()
				self.history_army_cnt_lst = rc_flat.copy()
				self.history.append(res_data)
				if stat:
					history_hash = self.save_history()

	def sendmap(self, stat):
		if self.numpy_fast_path():
			self.sendmap_numpy(stat)
			return
		history_hash = None
		dx = [0, -1, 1, 0, 0, -1, -1, 1, 1]
		dy = [0, 0, 0, -1, 1, -1, 1, -1, 1]
		kls = self.recentkills
		self.recentkills = {}
		leaderboard = self.build_leaderboard()
		for p in range(-1, self.pcnt):
			if p == -1 or self.watching[p]:
				rt = [[0 for j in range(self.m)] for i in range(self.n)]
				rc = [[0 for j in range(self.m)] for i in range(self.n)]
				for i in range(self.n):
					for j in range(self.m):
						if stat or p == -1 or self.team[p] == 0 or self.spec[p]:
							rt[i][j] = 200
						else:
							rt[i][j] = 202
							for d in range(9):
								if self.chkxy(i + dx[d], j + dy[d]) and self.owner[i + dx[d]][j + dy[d]] != 0 and self.team[self.owner[i + dx[d]][j + dy[d]] - 1] == self.team[p]:
									rt[i][j] = 200
						rc[i][j] = self.army_cnt[i][j] if rt[i][j] == 200 else 0
				for i in range(self.n):
					for j in range(self.m):
						if self.grid_type[i][j] == 2:
							rt[i][j] = 205 if rt[i][j] == 202 else 204 if self.owner[i][j] == 0 else self.owner[i][j] + 150
						elif self.grid_type[i][j] == 1:
							rt[i][j] = 201 if rt[i][j] == 200 else 203
						elif self.grid_type[i][j] == -1:
							rt[i][j] = self.owner[i][j] + 50 if rt[i][j] == 200 else 203
						elif self.grid_type[i][j] == -2:
							rt[i][j] = self.owner[i][j] + 100 if rt[i][j] == 200 else 202
						elif self.grid_type[i][j] == 0:
							rt[i][j] = 202 if rt[i][j] == 202 else self.owner[i][j] if self.owner[i][j] or self.army_cnt[i][j] else 200
				rt2 = []
				rc2 = []
				for i in range(self.n):
					for j in range(self.m):
						rt2.append(rt[i][j])
						rc2.append(rc[i][j])
				tmp = self.lst_move[p]
				if p == -1:
					force_full = not self.history or self.turn % 50 == 0 or stat
					prev_grid_type = self.history_grid_type_lst
					prev_army_cnt = self.history_army_cnt_lst
				else:
					force_full = stat or self.turn % 50 == 0 or random.randint(0, 50) == 0
					prev_grid_type = self.grid_type_lst[p]
					prev_army_cnt = self.army_cnt_lst[p]
				if force_full:
					res_data = {
						'grid_type': rt2,
						'army_cnt': rc2,
						'lst_move': {'x': tmp[0], 'y': tmp[1], 'dx': tmp[2], 'dy': tmp[3], 'half': tmp[4]},
						'leaderboard': leaderboard,
						'turn': self.turn,
						'kills': kls,
						'game_end': stat,
						'is_diff': False,
					}
				else:
					res_data = {
						'grid_type': get_diff(rt2, prev_grid_type),
						'army_cnt': get_diff(rc2, prev_army_cnt),
						'lst_move': {'x': tmp[0], 'y': tmp[1], 'dx': tmp[2], 'dy': tmp[3], 'half': tmp[4]},
						'leaderboard': leaderboard,
						'turn': self.turn,
						'kills': kls,
						'game_end': stat,
						'is_diff': True,
					}
				if history_hash:
					res_data['replay'] = history_hash
				if p != -1:
					self.grid_type_lst[p] = rt2
					self.army_cnt_lst[p] = rc2
					self.lst_move[p] = (-1, -1, -1, -1, False)
					self.update(self.player_ids[p], res_data)
				else:
					self.history_grid_type_lst = rt2
					self.history_army_cnt_lst = rc2
					self.history.append(res_data)
					if stat:
						history_hash = self.save_history()

	def add_move(self, player, x, y, dx, dy, half):
		player = self.player_ids_rev[player]
		if self.pstat[player] == left_game:
			return
		self.lock.acquire()
		self.pmove[player].append((x, y, dx, dy, half))
		self.lock.release()

	def clear_queue(self, player):
		player = self.player_ids_rev[player]
		self.lock.acquire()
		self.pmove[player] = []
		self.lock.release()

	def pop_queue(self, player):
		player = self.player_ids_rev[player]
		self.lock.acquire()
		if len(self.pmove[player]):
			self.pmove[player].pop()
		self.lock.release()

	def surrender(self, sid):
		player = self.player_ids_rev[sid]
		self.lock.acquire()
		if self.pstat[player] == 0 and self.surrender_turn[player] == 0:
			self.surrender_turn[player] = self.turn + surrender_delay_turns
			self.pmove[player] = []
			self.pstat[player] = left_game
			self.deadcount += 1
			self.deadorder[player] = self.deadcount
			self.spec[player] = True
			self.send_system_message(self.names[player] + ' will surrender in ' + str(surrender_delay_turns // 2) + ' turns.')
		self.lock.release()

	def apply_surrender(self, player):
		p = player + 1
		if self.numpy_fast_path():
			owned = self.owner == p
			self.owner[owned] = 0
			self.grid_type[owned & (self.grid_type == -2)] = -1
		else:
			for i in range(self.n):
				for j in range(self.m):
					if self.owner[i][j] == p:
						self.owner[i][j] = 0
						if self.grid_type[i][j] == -2:
							self.grid_type[i][j] = -1
		self.generals[player] = (-1, -1)
		self.surrender_turn[player] = 0
		self.send_system_message(self.names[player] + ' surrendered.')

	def kill(self, a, b):
		was_alive = self.pstat[b - 1] != left_game
		b_general = self.generals[b - 1] if b > 0 else (-1, -1)
		if self.numpy_fast_path():
			owned = self.owner == b
			self.owner[owned] = a
			self.army_cnt[owned] = (self.army_cnt[owned] + 1) // 2
			self.grid_type[owned & (self.grid_type == -2)] = -1
		else:
			for i in range(self.n):
				for j in range(self.m):
					if self.owner[i][j] == b:
						self.owner[i][j] = a
						self.army_cnt[i][j] = (self.army_cnt[i][j] + 1) // 2
						if self.grid_type[i][j] == -2:
							self.grid_type[i][j] = -1
		if self.move_general_on_capture and a > 0 and b > 0 and b_general != (-1, -1):
			a_general = self.generals[a - 1]
			if a_general != (-1, -1):
				self.grid_type[a_general[0]][a_general[1]] = -1
			self.grid_type[b_general[0]][b_general[1]] = -2
			self.generals[a - 1] = b_general
		self.generals[b - 1] = (-1, -1)
		self.surrender_turn[b - 1] = 0
		if was_alive:
			self.pstat[b - 1] = left_game
			self.deadcount += 1
			self.deadorder[b - 1] = self.deadcount
		self.spec[b - 1] = True
		if a > 0 and b > 0:
			self.recentkills[self.md5(self.player_ids[b - 1])] = self.names[a - 1]
			self.send_system_message(self.names[a - 1] + ' captured ' + self.names[b - 1] + '.')

	def chkmove(self, x, y, dx, dy, p):
		return self.chkxy(x, y) and self.chkxy(dx, dy) and abs(x - dx) + abs(y - dy) == 1 and self.owner[x][y] == p + 1 and self.army_cnt[x][y] > 0 and self.grid_type[dx][dy] != 1

	def attack(self, x, y, dx, dy, half):
		cnt = self.army_cnt[x][y] - 1
		if half:
			cnt //= 2
		self.army_cnt[x][y] -= cnt
		if self.owner[dx][dy] == self.owner[x][y]:
			self.army_cnt[dx][dy] += cnt
		elif self.owner[dx][dy] > 0 and self.owner[x][y] > 0 and self.team[int(self.owner[dx][dy]) - 1] == self.team[int(self.owner[x][y]) - 1]:
			self.army_cnt[dx][dy] += cnt
			if self.grid_type[dx][dy] != -2:
				self.owner[dx][dy] = self.owner[x][y]
		else:
			if cnt <= self.army_cnt[dx][dy]:
				self.army_cnt[dx][dy] -= cnt
			else:
				tmp = cnt - self.army_cnt[dx][dy]
				if self.grid_type[dx][dy] == -2:
					attacker = int(self.owner[x][y])
					defender = int(self.owner[dx][dy])
					self.kill(attacker, defender)
					self.grid_type[dx][dy] = -2 if self.move_general_on_capture and attacker > 0 else -1
				self.army_cnt[dx][dy] = tmp
				self.owner[dx][dy] = self.owner[x][y]

	def apply_natural_growth(self):
		if self.numpy_fast_path():
			if self.turn % 2 == 0:
				owned = self.owner > 0
				structures = (self.grid_type < 0) & owned
				swamps = (self.grid_type == 2) & owned
				self.army_cnt[structures] += 1
				self.army_cnt[swamps] -= 1
				self.owner[swamps & (self.army_cnt == 0)] = 0
			if self.turn % 50 == 0:
				self.army_cnt[self.owner > 0] += 1
			return
		if self.turn % 2 == 0:
			for i in range(self.n):
				for j in range(self.m):
					if self.grid_type[i][j] < 0 and self.owner[i][j] > 0:
						self.army_cnt[i][j] += 1
					elif self.grid_type[i][j] == 2 and self.owner[i][j] > 0:
						self.army_cnt[i][j] -= 1
						if self.army_cnt[i][j] == 0:
							self.owner[i][j] = 0
		if self.turn % 50 == 0:
			for i in range(self.n):
				for j in range(self.m):
					if self.owner[i][j] > 0:
						self.army_cnt[i][j] += 1

	def game_tick(self):
		self.turn += 1
		self.apply_natural_growth()
		for p in range(self.pcnt):
			if self.surrender_turn[p] and self.turn >= self.surrender_turn[p]:
				self.apply_surrender(p)
			if self.pstat[p]:
				self.pstat[p] = min(self.pstat[p] + 1, left_game)
				if self.pstat[p] == left_game - 1:
					self.kill(0, p + 1)
		tmp = range(self.pcnt)
		if self.turn % 2 == 1:
			tmp = list(reversed(tmp))
		self.lock.acquire()
		for p in tmp:
			while len(self.pmove[p]):
				mv = self.pmove[p].pop(0)
				if not self.chkmove(mv[0], mv[1], mv[2], mv[3], p):
					continue
				self.attack(mv[0], mv[1], mv[2], mv[3], mv[4])
				self.lst_move[p] = (mv[0], mv[1], mv[2], mv[3], mv[4])
				break
		self.lock.release()
		alive_team = {}
		for p in tmp:
			if self.pstat[p] != left_game:
				alive_team[self.team[p]] = True
		stat = len(alive_team) <= 1
		self.sendmap(stat)
		return stat

	def leave_game(self, sid):
		id = self.player_ids_rev[sid]
		if self.pstat[id] == 0:
			self.pstat[id] = 1
		self.watching[id] = False
		self.send_system_message(self.names[id] + ' left.')

	def save_history(self):
		os.makedirs('replays', exist_ok=True)
		res = {
			'n': self.n,
			'm': self.m,
			'history': self.history,
		}
		s = json.dumps(res, separators=(',', ':'))
		hs = base64.b64encode(hashlib.sha256(s.encode()).digest()[:9]).decode().replace('/', '-')
		open('replays/' + hs + '.json', 'w', encoding='utf-8').write(s)
		ranks = [x['uid']for x in sorted(self.history[-1]['leaderboard'], key=lambda x: x['dead'] + x['land'] * 100 + x['army'] * 10000000, reverse=True)]
		u = json.dumps({
			'time': int(time.time()),
			'id': hs,
			'rank': ranks,
			'turn': self.history[-1]['turn'] // 2,
		}) + '\n'
		open('replays/all.txt', 'a', encoding='utf-8').write(u)
		return hs

	def game_loop(self):
		eventlet.sleep(max(0.01, self.otime + 2 - time.time()))
		lst = time.time()
		self.sendmap(False)
		self.announce_map_generation()
		while True:
			eventlet.sleep(max(0.01, 0.5 / self.speed - time.time() + lst))
			lst = time.time()
			if self.game_tick():
				break
		res = ''
		for p in range(self.pcnt):
			if self.pstat[p] != left_game:
				if res != '':
					res += ','
				res += self.names[p]
		print('end game', self.gid)
		sys.stdout.flush()
		self.send_system_message(res + ' win.')
		self.end_game(self.gid)

	def start_game(self, socketio):
		socketio.start_background_task(target=self.game_loop)

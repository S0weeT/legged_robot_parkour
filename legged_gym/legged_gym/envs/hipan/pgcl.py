import numpy as np
import heapq


class PGCLManager:
    """Path-Guided Curriculum Learning: progressively removes intermediate subgoals
    along a global A* path, extending navigation horizon."""

    def __init__(self, initial_d=1.0, d_step=1.0, path_planner=None):
        self.initial_d = initial_d
        self.d_step = d_step
        self.current_d = initial_d
        self.level = 0
        self.path_planner = path_planner

    def generate_subgoals(self, start, goal, privileged_map):
        """Place subgoals every d meters along global path."""
        if self.path_planner is None:
            return [goal]

        path = self.path_planner.plan(start, goal, privileged_map)
        if len(path) < 2:
            return [goal]

        subgoals = []
        accumulated_dist = 0.0
        last_point = path[0]
        subgoals.append(last_point)

        for point in path[1:]:
            seg_dist = np.linalg.norm(np.array(point) - np.array(last_point))
            accumulated_dist += seg_dist
            last_point = point
            if accumulated_dist >= self.current_d:
                subgoals.append(point)
                accumulated_dist = 0.0

        if subgoals[-1] != goal:
            subgoals.append(goal)
        return subgoals

    def advance_level(self):
        """All subgoals reached -> increase d -> harder."""
        self.current_d += self.d_step
        self.level += 1
        return self.current_d

    def is_final_level(self, path_length):
        """Final level: d > total path length -> only start->goal."""
        return self.current_d > path_length

    def get_state_dict(self):
        return {'level': self.level, 'current_d': self.current_d}


class AStarPlanner:
    """A* path planner on 2D occupancy grid from height map."""

    def __init__(self, resolution=0.1, obstacle_threshold=0.3):
        self.resolution = resolution
        self.obstacle_threshold = obstacle_threshold

    def plan(self, start, goal, height_map):
        """Plan path from start [x,y] to goal [x,y] on height_map (H,W)."""
        h, w = height_map.shape
        start_rc = self._world_to_grid(start, h, w)
        goal_rc = self._world_to_grid(goal, h, w)

        grad_y, grad_x = np.gradient(height_map)
        obstacle_mask = (np.abs(grad_y) + np.abs(grad_x)) > self.obstacle_threshold

        open_set = [(0, start_rc)]
        came_from = {}
        g_score = {start_rc: 0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_rc:
                path = [goal]
                while current in came_from:
                    current = came_from[current]
                    path.append(self._grid_to_world(current, h, w))
                path.reverse()
                return path

            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                neighbor = (current[0]+dr, current[1]+dc)
                if not (0 <= neighbor[0] < h and 0 <= neighbor[1] < w):
                    continue
                if obstacle_mask[neighbor]:
                    continue
                tentative_g = g_score[current] + (1.414 if dr != 0 and dc != 0 else 1.0)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal_rc)
                    heapq.heappush(open_set, (f_score, neighbor))
                    came_from[neighbor] = current
        return [start, goal]  # no path found, direct line

    def _world_to_grid(self, pos, h, w):
        r = int(pos[1] / self.resolution)
        c = int(pos[0] / self.resolution)
        return (np.clip(r, 0, h-1), np.clip(c, 0, w-1))

    def _grid_to_world(self, grid, h, w):
        return [grid[1] * self.resolution, grid[0] * self.resolution, 0.0]

    def _heuristic(self, a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


class StateCountExploration:
    """Discrete-grid state visit counter for Intrinsic Reward (1/sqrt(count))."""

    def __init__(self, resolution=0.1, grid_size=(200, 200)):
        self.resolution = resolution
        self.counts = np.zeros(grid_size)

    def get_count(self, pos_xy):
        idx = self._pos_to_idx(pos_xy)
        return self.counts[idx]

    def increment(self, pos_xy):
        idx = self._pos_to_idx(pos_xy)
        self.counts[idx] += 1

    def get_intrinsic_reward(self, pos_xy):
        count = self.get_count(pos_xy)
        if count < 1:
            return 1.0
        return 1.0 / np.sqrt(count)

    def _pos_to_idx(self, pos_xy):
        x_idx = int(pos_xy[0] / self.resolution)
        y_idx = int(pos_xy[1] / self.resolution)
        x_idx = np.clip(x_idx, 0, self.counts.shape[1] - 1)
        y_idx = np.clip(y_idx, 0, self.counts.shape[0] - 1)
        return (y_idx, x_idx)

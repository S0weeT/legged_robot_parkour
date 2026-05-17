import numpy as np
from isaacgym.terrain_utils import (
    SubTerrain, convert_heightfield_to_trimesh,
    sloped_terrain, pyramid_sloped_terrain, random_uniform_terrain,
    discrete_obstacles_terrain, wave_terrain, stairs_terrain,
)


class WFCTerrain:
    """WFC-style terrain: randomly arranges tile types to produce unstructured 3D terrain."""

    TILE_TYPES = ['flat', 'rough_slope', 'rough', 'obstacles', 'wave',
                  'stairs_up', 'stairs_down', 'pillars', 'mixed']

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_rows = cfg.num_rows
        self.num_cols = cfg.num_cols
        self.tile_width = int(cfg.terrain_length / cfg.horizontal_scale / cfg.num_rows)
        self.tile_length = self.tile_width
        self.horizontal_scale = cfg.horizontal_scale
        self.vertical_scale = cfg.vertical_scale

        self.border = int(cfg.border_size / cfg.horizontal_scale)
        self.tot_rows = int(self.num_rows * self.tile_length) + 2 * self.border
        self.tot_cols = int(self.num_cols * self.tile_width) + 2 * self.border

        self.heightsamples = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self._generate()
        self.vertices, self.triangles = convert_heightfield_to_trimesh(
            self.heightsamples,
            horizontal_scale=self.horizontal_scale,
            vertical_scale=self.vertical_scale,
            slope_threshold=1.5,
        )

    def _new_sub_terrain(self):
        return SubTerrain(width=self.tile_width, length=self.tile_length,
                          vertical_scale=self.vertical_scale,
                          horizontal_scale=self.horizontal_scale)

    def _generate_tile(self, tile_type, terrain):
        if tile_type == 'flat':
            return sloped_terrain(terrain, slope=0.0).height_field_raw
        elif tile_type == 'rough_slope':
            slope = np.random.uniform(-0.5, 0.5)
            return pyramid_sloped_terrain(terrain, slope=slope).height_field_raw
        elif tile_type == 'rough':
            return random_uniform_terrain(terrain, min_height=-0.1,
                                          max_height=0.1, step=0.15,
                                          downsampled_scale=0.5).height_field_raw
        elif tile_type == 'obstacles':
            return discrete_obstacles_terrain(terrain, max_height=0.2,
                                              min_size=0.5, max_size=3.,
                                              num_rects=15).height_field_raw
        elif tile_type == 'wave':
            return wave_terrain(terrain, num_waves=2.,
                                amplitude=np.random.uniform(0.5, 1.5)).height_field_raw
        elif tile_type == 'stairs_up':
            return stairs_terrain(terrain, step_width=0.5,
                                  step_height=np.random.uniform(0.05, 0.2)).height_field_raw
        elif tile_type == 'stairs_down':
            return stairs_terrain(terrain, step_width=0.5,
                                  step_height=-0.15).height_field_raw
        elif tile_type == 'pillars':
            return discrete_obstacles_terrain(terrain, max_height=0.4,
                                              min_size=0.3, max_size=0.5,
                                              num_rects=30).height_field_raw
        elif tile_type == 'mixed':
            return random_uniform_terrain(terrain, min_height=-0.2,
                                          max_height=0.3, step=0.2,
                                          downsampled_scale=0.4).height_field_raw
        else:
            return sloped_terrain(terrain, slope=0.0).height_field_raw

    def _generate(self):
        n_tiles = self.num_rows * self.num_cols
        tiles = np.random.choice(self.TILE_TYPES, size=n_tiles, replace=True)
        for idx, tile_type in enumerate(tiles):
            row = idx // self.num_cols
            col = idx % self.num_cols
            terrain = self._new_sub_terrain()
            hf = self._generate_tile(tile_type, terrain)
            r_start = self.border + int(row * self.tile_length)
            c_start = self.border + int(col * self.tile_width)
            self.heightsamples[r_start:r_start + self.tile_length,
                               c_start:c_start + self.tile_width] = hf

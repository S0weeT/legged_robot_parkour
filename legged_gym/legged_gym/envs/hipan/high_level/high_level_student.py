# legged_gym/envs/hipan/high_level/high_level_student.py
"""HiPAN high-level student policy (placeholder — implemented in a later task)."""

from legged_gym.envs.hipan.high_level.high_level_teacher import HighLevelTeacher


class HighLevelStudent(HighLevelTeacher):
    """HiPAN high-level student: deploys without privileged map access.

    Full implementation in a subsequent task replaces the privileged M_3D / M_2.5D
    perception with domain-randomized sensor-like observations and distilled behavior.
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

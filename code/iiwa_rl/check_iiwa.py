# Ucita iiwa7 USD i pusti simulaciju za vizualnu provjeru uvoza.
import os, argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IIWA_USD = os.path.join(REPO_ROOT, "assets", "iiwa7.usd")


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.01, device="cpu"))
    sim.set_camera_view(eye=[2.0, 2.0, 2.0], target=[0.0, 0.0, 0.5])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0)
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = sim_utils.UsdFileCfg(usd_path=IIWA_USD)
    robot_cfg.func("/World/iiwa", robot_cfg)

    sim.reset()
    print(f"=== iiwa ucitan iz {IIWA_USD}. Zatvori prozor za kraj. ===")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()

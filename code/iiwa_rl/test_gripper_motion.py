"""
Pokretanje:
    ./isaaclab.sh -p test_gripper_motion.py --usd_path output/kmr_iiwa_full.usd
"""

import argparse

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--usd_path", type=str, required=True, help="Putanja do kmr_iiwa_full.usd"
)
parser.add_argument("--headless", action="store_true", help="Pokreni bez GUI-ja")
parser.add_argument(
    "--articulation_prim_path",
    type=str,
    default="/kmr_iiwa/base_link",
    help="Prim path root artikulacije (isti kao u cmd_vel_bridge.py - gripper zglobovi su "
    "dio ISTE artikulacije kao ruka, ne treba zaseban Articulation objekt).",
)
parser.add_argument(
    "--cycle_seconds",
    type=float,
    default=2.0,
    help="Vrijeme za jedan otvoren->zatvoren prijelaz.",
)
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

import numpy as np  # noqa: E402

import isaacsim.core.utils.stage as stage_utils  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import Articulation  # noqa: E402

FINGER_JOINTS = [f"gripper_finger_{i}_joint" for i in range(1, 5)]
STROKE = 0.025  # m, iz URDF limita (0=otvoren, 0.025=zatvoren)


def main():
    stage_utils.open_stage(args.usd_path)

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    world.reset()

    robot = Articulation(prim_paths_expr=args.articulation_prim_path, name="kmr_iiwa")
    world.scene.add(robot)
    world.reset()

    dof_names = robot.dof_names
    print("[INFO] Svi DOF nazivi:", dof_names)
    try:
        finger_indices = np.array([dof_names.index(n) for n in FINGER_JOINTS])
    except ValueError as e:
        print(f"[GREŠKA] Ne mogu naći finger zglob u dof_names: {e}")
        print(
            "         Usporedi točne nazive iznad s FINGER_JOINTS listom u ovom fileu."
        )
        simulation_app.close()
        return
    print("[INFO] finger DOF indeksi:", finger_indices.tolist())

    dt = world.get_physics_dt()
    steps_per_cycle = max(1, int(args.cycle_seconds / dt))
    step = 0

    print("[INFO] Ciklički otvaranje/zatvaranje. Ctrl+C za izlaz.")
    try:
        while simulation_app.is_running():
            phase = (step % (2 * steps_per_cycle)) / steps_per_cycle  # 0..2
            closing_fraction = (
                phase if phase <= 1.0 else 2.0 - phase
            )  # trokutasti val 0->1->0
            target = np.full((1, len(finger_indices)), closing_fraction * STROKE)

            robot.set_joint_position_targets(target, joint_indices=finger_indices)
            world.step(render=not args.headless)

            if step % 30 == 0:
                current = robot.get_joint_positions(joint_indices=finger_indices)
                print(
                    f"[step {step:5d}] target={closing_fraction * STROKE:.4f}  stvarno={np.round(current[0], 4)}"
                )
            step += 1
    except KeyboardInterrupt:
        pass
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

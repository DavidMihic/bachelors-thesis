"""Prošireni URDF->USD converter za kmr_iiwa + gripper.

Isto kao IsaacLab scripts/tools/convert_urdf.py, ali dodaje --collider-type
i --self-collision zastavice koje stock CLI ne izlaže (UrdfConverterCfg ih
podržava, samo nisu wireane u argparse gornjeg sloja).

Pokretanje (primjer):
  ./isaaclab.sh -p convert_gripper_urdf.py \
      code/ros2_ws/src/kmr_iiwa_description/urdf/kmr_iiwa_full.urdf \
      output/kmr_iiwa_full.usd \
      --collider-type convex_decomposition \
      --fix-base

NAPOMENA: --merge-joints namjerno NIJE dodan kao skraćenica na "sigurno";
zastavica postoji ali default ostaje False. Ne uključuj je za ovaj robot —
merge bi obrisao gripper_wrist_joint (fixed) na kojem se čita F/T.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="URDF -> USD s punom kontrolom collision postavki.")
parser.add_argument("input", type=str, help="Putanja do ulaznog URDF-a.")
parser.add_argument("output", type=str, help="Putanja gdje spremiti USD.")
parser.add_argument("--fix-base", action="store_true", default=False,
                     help="Fiksiraj root link (koristi za standalone gripper test, NE za puni robot na omniMove bazi).")
parser.add_argument("--merge-joints", action="store_true", default=False,
                     help="Spaja fixed jointove. OPREZ: briše gripper_wrist_joint (F/T senzor) ako je uključeno.")
parser.add_argument("--collider-type", type=str, default="convex_decomposition",
                     choices=["convex_hull", "convex_decomposition"],
                     help="convex_decomposition je OBAVEZAN za konkavnu geometriju kuke prsta.")
parser.add_argument("--self-collision", action="store_true", default=False,
                     help="Self-collision unutar articulation grafa. Default False (parent-child parovi su svejedno filtrirani).")
parser.add_argument("--make-instanceable", action="store_true", default=False,
                     help="USD scene-graph instancing (dijeli mesh izmedu 4 identicna prsta). "
                          "Default False jer instance proxy prime NIJE moguce direktno bindati "
                          "materijalom bez dodatnog resolvanja na prototype - vidi apply_gripper_friction.py.")
parser.add_argument("--arm-stiffness", type=float, default=100000.0,
                     help="Stiffness za iiwa_joint_1..7 (Nm/rad, prije internog deg-skaliranja). "
                          "Default 100000 rekonstruiran iz tvog postojećeg kmr_iiwa.usd (bio je 'kruta ruka').")
parser.add_argument("--arm-damping", type=float, default=5000.0,
                     help="Damping za iiwa_joint_1..7. Default 5000, isto rekonstruirano iz kmr_iiwa.usd.")
parser.add_argument("--arm-joint-pattern", type=str, default=r"^iiwa_joint_[1-7]$",
                     help="Regex koji pogađa imena arm zglobova u URDF-u.")
parser.add_argument("--gripper-stiffness", type=float, default=2000.0,
                     help="Stiffness za gripper_finger_1..4_joint (N/m). NIJE rekonstruirano iz postojećeg "
                          "filea kao arm vrijednosti - ovo je razuman početni pogodak za 25mm hod / "
                          "30N effort limit, treba ga potvrditi promatranjem stiska u simulaciji.")
parser.add_argument("--gripper-damping", type=float, default=100.0,
                     help="Damping za gripper_finger_1..4_joint. Isto - početni pogodak, ne izmjerena vrijednost.")
parser.add_argument("--gripper-joint-pattern", type=str, default=r"^gripper_finger_[1-4]_joint$",
                     help="Regex koji pogađa imena gripper prismatic zglobova u URDF-u.")
parser.add_argument("--joint-target-type", type=str, default="position", choices=["position", "velocity", "none"])

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from isaaclab.utils.assets import check_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402


def main():
    urdf_path = os.path.abspath(args_cli.input)
    if not check_file_path(urdf_path):
        raise ValueError(f"Invalid file path: {urdf_path}")
    dest_path = os.path.abspath(args_cli.output)

    if args_cli.merge_joints:
        print("!! UPOZORENJE: --merge-joints je uključen — gripper_wrist_joint (F/T) ce biti spojen i nestat ce.")

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        fix_base=args_cli.fix_base,
        merge_fixed_joints=args_cli.merge_joints,
        force_usd_conversion=True,
        collider_type=args_cli.collider_type,
        self_collision=args_cli.self_collision,
        make_instanceable=args_cli.make_instanceable,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness={
                    args_cli.arm_joint_pattern: args_cli.arm_stiffness,
                    args_cli.gripper_joint_pattern: args_cli.gripper_stiffness,
                },
                damping={
                    args_cli.arm_joint_pattern: args_cli.arm_damping,
                    args_cli.gripper_joint_pattern: args_cli.gripper_damping,
                },
            ),
            target_type=args_cli.joint_target_type,
        ),
    )

    print("-" * 80)
    print(f"Input URDF: {urdf_path}")
    print_dict(cfg.to_dict(), nesting=0)
    print("-" * 80)

    converter = UrdfConverter(cfg)
    print(f"Generated USD file: {converter.usd_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()

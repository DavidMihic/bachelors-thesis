"""Upisuje pocetne kutove zglobova ruke u USD, POSLIJE URDF->USD konverzije.

Bez ovoga se robot radja sa svim zglobovima na nuli, dakle s rukom uspravno
ispred sebe. To je blizu singularnosti, zaklanja kameru i tagove na vratima, i
lose izgleda na slikama. Rjesenje slanjem naredbe nakon pokretanja ne pomaze
do kraja - ruka svejedno na trenutak stoji uspravno dok se naredba ne izvrsi.

Postavlja se i pocetno stanje zgloba (PhysicsJointStateAPI, position) i cilj
pogona (drive targetPosition), jer bi inace pogon odmah povukao zglob natrag
na svoj stari cilj.

Kutovi su u RADIJANIMA na ulazu; USD angular drive i joint state su u
stupnjevima pa se pretvaraju.

Pokretanje:
  ./isaaclab.sh -p apply_arm_home_pose.py assets/configuration/kmr_iiwa_full.usd
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Upisi pocetne kutove zglobova ruke u USD."
)
parser.add_argument(
    "usd_path", type=str, help="Putanja do USD filea (mijenja se in-place)."
)
parser.add_argument(
    "--joint-prefix",
    type=str,
    default="iiwa_joint_",
    help="Prefiks imena zglobova ruke.",
)
parser.add_argument(
    "--pose",
    type=float,
    nargs=7,
    default=[0.0, -0.6, 0.0, -2.2, 0.0, 0.8, 0.0],
    metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
    help="Kutovi zglobova 1-7 u radijanima. Default drzi ruku skupljenu iznad "
    "baze, izvan vidnog polja kamere.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics, PhysxSchema  # noqa: E402


def main():
    stage = Usd.Stage.Open(args_cli.usd_path)
    if stage is None:
        raise ValueError(f"Ne mogu otvoriti USD: {args_cli.usd_path}")

    targets = {f"{args_cli.joint_prefix}{i+1}": args_cli.pose[i] for i in range(7)}

    # Obidji i instance proxije, pa autoriraj u backing layeru - isti razlog
    # kao u apply_gripper_friction.py.
    written, missing = [], set(targets)
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        name = prim.GetName()
        if name not in targets:
            continue

        source = prim.GetPrimInPrototype() if prim.IsInstanceProxy() else prim
        angle_deg = math.degrees(targets[name])

        # 1. pocetno stanje zgloba
        state_api = PhysxSchema.JointStateAPI.Apply(source, "angular")
        state_api.CreatePositionAttr().Set(angle_deg)
        state_api.CreateVelocityAttr().Set(0.0)

        # 2. cilj pogona - inace pogon odmah povuce zglob natrag na stari cilj
        drive = UsdPhysics.DriveAPI.Get(source, "angular")
        if drive:
            drive.CreateTargetPositionAttr().Set(angle_deg)

        written.append(f"{name} = {angle_deg:+.2f} deg")
        missing.discard(name)

    if not written:
        raise RuntimeError(
            f"Nijedan zglob s prefiksom '{args_cli.joint_prefix}' nije nadjen. "
            "Provjeri putanju i nazive zglobova."
        )

    for line in written:
        print("  ", line)
    if missing:
        print("  UPOZORENJE, nisu nadjeni:", ", ".join(sorted(missing)))

    stage.Save()
    print(f"Spremljeno u {args_cli.usd_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()

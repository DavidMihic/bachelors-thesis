"""
check.py — provjera kmr_iiwa.usd u Isaac Simu.

Pokretanje (iz IsaacLab root direktorija):
    ./isaaclab.sh -p check.py --usd_path <putanja>/assets/kmr_iiwa.usd

Ili s obicnim Isaac Sim python.sh:
    ./python.sh check.py --usd_path <putanja>/assets/kmr_iiwa.usd

Sto radi:
  1. Otvara stage i ispisuje sve linkove/jointove artikulacije (provjera da
     nista nije nestalo/duplo nakon merge-joints konverzije).
  2. Ispisuje mount offset (base_link -> iiwa_link_0) da vizualno/brojcano
     potvrdis da je ruka na krovu baze, ne u zraku/unutar baze.
  3. Dodaje ground plane i pokrece par koraka fizike bez ikakve kontrole
     jointova — ako se robot "raspadne"/eksplodira, znak je da negdje
     imas self-collision ili krivu inerciju/collision geometriju.
  4. Ostaje otvoren (GUI) da mozes rucno pogledati scenu.
"""

import argparse

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--usd_path", type=str, required=True, help="Putanja do kmr_iiwa.usd")
parser.add_argument("--headless", action="store_true", help="Pokreni bez GUI-ja")
parser.add_argument("--sim_steps", type=int, default=120, help="Broj fizickih koraka za sanity-check pad pod gravitacijom")
args = parser.parse_args()

simulation_app = SimulationApp({"headless": args.headless})

# Imports koji zahtijevaju da je SimulationApp vec pokrenut
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics
import isaacsim.core.utils.stage as stage_utils
import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation


def print_stage_structure(stage: Usd.Stage):
    print("\n" + "=" * 70)
    print("LINKOVI I JOINTOVI U STAGE-u")
    print("=" * 70)
    for prim in stage.Traverse():
        prim_type = prim.GetTypeName()
        if prim_type in ("PhysicsFixedJoint", "PhysicsRevoluteJoint", "PhysicsPrismaticJoint"):
            joint_api = UsdPhysics.Joint(prim)
            body0_rel = joint_api.GetBody0Rel().GetTargets()
            body1_rel = joint_api.GetBody1Rel().GetTargets()
            parent = body0_rel[0] if body0_rel else "?"
            child = body1_rel[0] if body1_rel else "?"
            print(f"  [JOINT:{prim_type.replace('Physics', '').replace('Joint','')}] "
                  f"{prim.GetPath()}  ({parent} -> {child})")
        elif prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            tags = []
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                tags.append("ARTICULATION_ROOT")
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                tags.append("RIGID_BODY")
            print(f"  [LINK] {prim.GetPath()}  {tags}")
    print("=" * 70 + "\n")


def print_mount_offset(stage: Usd.Stage, base_prim_path="/kmr_iiwa/base_link", iiwa_prim_path="/kmr_iiwa/iiwa_link_1"):
    base_prim = stage.GetPrimAtPath(base_prim_path)
    iiwa_prim = stage.GetPrimAtPath(iiwa_prim_path)
    if not base_prim.IsValid() or not iiwa_prim.IsValid():
        print(f"[WARN] Nisam nasao ocekivane prim pathove ({base_prim_path}, {iiwa_prim_path}) "
              f"- provjeri stvarne nazive u stageu gore i azuriraj ovu funkciju.")
        return
    base_xform = UsdGeom.Xformable(base_prim)
    iiwa_xform = UsdGeom.Xformable(iiwa_prim)
    base_world = base_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    iiwa_world = iiwa_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    base_pos = base_world.ExtractTranslation()
    iiwa_pos = iiwa_world.ExtractTranslation()
    offset = iiwa_pos - base_pos
    print(f"[MOUNT OFFSET] base_link world pos: {base_pos}")
    print(f"[MOUNT OFFSET] iiwa_link_0 world pos: {iiwa_pos}")
    print(f"[MOUNT OFFSET] iiwa relative to base: {offset}")
    print("  -> Provjeri odgovara li ovo stvarnoj mount tocki na krovu baze.\n")


def main():
    # Ucitaj stage
    stage_utils.open_stage(args.usd_path)
    stage = omni.usd.get_context().get_stage()

    print_stage_structure(stage)
    print_mount_offset(stage)

    # Postavi World s ground planeom za fizicki sanity-check
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # Referenciraj robota u scenu ako vec nije root (ako je USD sam robot,
    # ovo samo osigurava da postoji ArticulationView za praćenje)
    world.reset()

    try:
        robot = Articulation(prim_paths_expr="/kmr_iiwa/base_link", name="kmr_iiwa")
        world.scene.add(robot)
        world.reset()
        print(f"[OK] Artikulacija ucitana, broj DOF-ova: {robot.num_dof}")
        print(f"[OK] Nazivi jointova: {robot.dof_names}")
    except Exception as e:
        print(f"[WARN] Nisam uspio automatski uhvatiti Articulation na /base_link: {e}")
        print("       Nastavljam samo s fizickom simulacijom bez eksplicitnog articulation handlea.")

    print(f"\n[SIM] Pokrecem {args.sim_steps} fizickih koraka bez kontrole jointova "
          f"(sanity-check da se robot ne raspadne pod gravitacijom)...")
    for i in range(args.sim_steps):
        world.step(render=not args.headless)

    print("[SIM] Gotovo. Ako je robot i dalje u jednom komadu i stoji na podu, geometrija/fizika je OK.")
    print("      Ako se raspao/eksplodirao, provjeri self-collision, collision meseve i inertial vrijednosti.")

    if not args.headless:
        print("\n[INFO] GUI ostaje otvoren - pogledaj scenu rucno, zatvori prozor kad zavrsis.")
        while simulation_app.is_running():
            world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()

"""
build_integration_scene.py - Gradi novu USD scenu koja REFERENCIRA gotove
assete (robot + vrata) i dodaje ground plane. NE dira robot ili vrata USD-ove -
samo ih slaze zajedno preko USD referenci.

ARHITEKTURNA NAMJERA: kmr_iiwa_full.usd ostaje "cist" robot asset (samo ono
sto convert_urdf.py + add_camera_ros_graph.py generiraju), vrata USD ostaje
"cist" door asset. Ova skripta pravi treci, zaseban fajl koji ih kombinira -
kad se robot ili vrata promijene (npr. re-konverzija nakon izmjene krutosti
zglobova), integracijska scena to automatski vidi kroz referencu, ne treba
je ponovno graditi.

Vrata imaju normalu (+X u vlastitom frameu, dogovoreno u revolute_door.urdf/
sliding_door.urdf) - default door-yaw=180 okrece ih da gledaju nazad prema
robotu koji dolazi iz +X smjera.

Pokretanje:
    ./isaaclab.sh -p build_integration_scene.py \
        --robot assets/kmr_iiwa_full.usd \
        --door assets/revolute_door.usd \
        --door-offset 2.5 0 0 --door-yaw 180 \
        --output assets/integration_test_revolute.usd
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Gradi integracijsku scenu (robot + vrata + ground plane) preko USD referenci."
)
parser.add_argument(
    "--robot",
    type=str,
    required=True,
    help="Putanja do robot USD-a (npr. kmr_iiwa_full.usd).",
)
parser.add_argument(
    "--door",
    type=str,
    required=True,
    help="Putanja do vrata USD-a (revolute_door.usd ili sliding_door.usd).",
)
parser.add_argument(
    "--output", type=str, required=True, help="Putanja izlazne integracijske scene."
)
parser.add_argument(
    "--door-offset",
    type=float,
    nargs=3,
    default=[2.5, 0.0, 0.0],
    metavar=("X", "Y", "Z"),
    help="Pozicija vrata relativno na robotov spawn (m). Default 2.5m ispred (+X).",
)
parser.add_argument(
    "--door-yaw",
    type=float,
    default=180.0,
    help="Rotacija vrata oko Z (stupnjevi). Default 180 - vratina +X normala gleda nazad prema robotu.",
)
parser.add_argument(
    "--skip-ground-plane",
    action="store_true",
    help="Ne dodaji ground plane (npr. ako ga vec ima neka druga referencirana scena).",
)

# --- Nasumican spawn baze ispred vrata ---
# ARHITEKTURNA NAPOMENA: robot ostaje na cistom originu (nema smisla dirati
# kmr_iiwa_full.usd niti ovu skriptinu "robot se referencira bez transformacije"
# granu) - randomiziramo UMJESTO TOGA relativni offset/yaw vrata. Rezultat je
# geometrijski identican (jedino relativna poza baza<->vrata ima fizikalno
# znacenje u praznoj sceni), a --door-offset/--door-yaw ostaju kompatibilni sa
# starim, ne-randomiziranim pozivima (default randomize=False).
parser.add_argument(
    "--randomize-spawn",
    action="store_true",
    help="Ako je postavljeno, --door-offset/--door-yaw se koriste kao BAZA na koju se "
    "dodaje nasumican pomak iz raspona ispod (umjesto da se koriste doslovno).",
)
parser.add_argument(
    "--spawn-seed",
    type=int,
    default=None,
    help="Seed za nasumicni generator (reproducibilnost). Default None = nedeterministicko.",
)
parser.add_argument(
    "--spawn-distance-range",
    type=float,
    nargs=2,
    default=[1.8, 3.2],
    metavar=("MIN", "MAX"),
    help="Raspon udaljenosti (m, X-offset vrata) unutar kojeg spawnamo. PLACEHOLDER — "
    "doc §3 trazi da ovo bude empirijski provjereno prema dometu detekcije "
    "door_tag_center (80mm tag) s postojecom D435 konfiguracijom kamere "
    "(add_camera_ros_graph.py), NE pretpostavljena vrijednost. Podesi nakon prvog testa.",
)
parser.add_argument(
    "--spawn-lateral-range",
    type=float,
    nargs=2,
    default=[-0.6, 0.6],
    metavar=("MIN", "MAX"),
    help="Raspon bocnog (Y) offseta vrata (m) - simulira lateralni pomak robota prema vratima.",
)
parser.add_argument(
    "--spawn-yaw-jitter-deg",
    type=float,
    default=20.0,
    help="+/- perturbacija oko --door-yaw (stupnjevi) - simulira kut prilaska koji nije "
    "dead-on, tako da P/PD regulator iz faze prilaska (§3) ima sto korigirati.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402


def sample_spawn(args) -> tuple[list[float], float, dict]:
    """Nasumicno izvuci (door_offset, door_yaw) oko zadane baze. Vraca i dict
    s upotrijebljenim vrijednostima/seedom radi logiranja (vidi main())."""
    rng = random.Random(args.spawn_seed)
    distance = rng.uniform(*args.spawn_distance_range)
    lateral = rng.uniform(*args.spawn_lateral_range)
    yaw_jitter = rng.uniform(-args.spawn_yaw_jitter_deg, args.spawn_yaw_jitter_deg)

    door_offset = [distance, lateral, args.door_offset[2]]
    door_yaw = args.door_yaw + yaw_jitter

    draw = {
        "seed": args.spawn_seed,
        "distance_m": distance,
        "lateral_m": lateral,
        "yaw_jitter_deg": yaw_jitter,
        "resulting_door_offset": door_offset,
        "resulting_door_yaw_deg": door_yaw,
    }
    return door_offset, door_yaw, draw


def add_ground_plane(stage: Usd.Stage, parent_path: str = "/World"):
    """Ista struktura kao Create > Physics > Ground Plane u Isaac Sim GUI-u -
    25x25m vizualna ravnina + beskonacna PhysX collision plane."""
    gp_path = f"{parent_path}/GroundPlane"
    gp = UsdGeom.Xform.Define(stage, gp_path)
    gp.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))
    gp.AddOrientOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1, 0, 0, 0))
    gp.AddScaleOp(precision=UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(1, 1, 1))

    mesh = UsdGeom.Mesh.Define(stage, f"{gp_path}/CollisionMesh")
    mesh.CreateDoubleSidedAttr(False)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([(0, 0, 1)] * 4)
    mesh.CreatePointsAttr([(-25, -25, 0), (25, -25, 0), (25, 25, 0), (-25, 25, 0)])
    mesh.CreateDisplayColorAttr([(0.5, 0.5, 0.5)])

    plane = UsdGeom.Plane.Define(stage, f"{gp_path}/CollisionPlane")
    plane.CreateAxisAttr("Z")
    plane.CreatePurposeAttr("guide")
    UsdPhysics.CollisionAPI.Apply(plane.GetPrim())

    print(f"Ground plane dodan na {gp_path}")


def add_physics_scene_if_missing(stage: Usd.Stage, path: str = "/World/PhysicsScene"):
    for prim in stage.Traverse():
        if prim.GetTypeName() == "PhysicsScene":
            print(f"PhysicsScene vec postoji na {prim.GetPath()}, ne dodajem novu")
            return
    UsdPhysics.Scene.Define(stage, path)
    print(f"PhysicsScene dodana na {path}")


def main():
    for label, path in [("robot", args_cli.robot), ("door", args_cli.door)]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} USD ne postoji: {path}")

    stage = Usd.Stage.CreateNew(args_cli.output)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # --- Robot: referenca na origin, bez dodatne transformacije ---
    robot_path = "/World/Robot"
    robot_prim = stage.DefinePrim(robot_path, "Xform")
    robot_prim.GetReferences().AddReference(
        os.path.relpath(args_cli.robot, os.path.dirname(args_cli.output))
    )
    print(f"Robot referenciran na {robot_path} -> {args_cli.robot}")

    # --- §3: nasumican spawn (randomizira relativni offset/yaw vrata, robot
    # ostaje neizmijenjen gore) - vidi sample_spawn() i napomenu uz argparse. ---
    if args_cli.randomize_spawn:
        door_offset, door_yaw, spawn_draw = sample_spawn(args_cli)
        print(f"[RANDOMIZE-SPAWN] izvuceno: {spawn_draw}")
        sidecar_path = args_cli.output + ".spawn.json"
        with open(sidecar_path, "w") as f:
            json.dump(spawn_draw, f, indent=2)
        print(
            f"[RANDOMIZE-SPAWN] zapisano u {sidecar_path} (za reprodukciju/analizu kasnije)"
        )
    else:
        door_offset, door_yaw = args_cli.door_offset, args_cli.door_yaw

    # --- Vrata: referenca na (eventualno randomiziranom) offsetu/rotaciji ---
    door_path = "/World/Door"
    door_prim = stage.DefinePrim(door_path, "Xform")
    door_prim.GetReferences().AddReference(
        os.path.relpath(args_cli.door, os.path.dirname(args_cli.output))
    )
    door_xform = UsdGeom.Xformable(door_prim)
    door_xform.ClearXformOpOrder()  # door asset vec ima svoje xformOps iz vlastite
    # konverzije - bez ovoga AddTranslateOp puca na
    # "already exists in xformOpOrder"
    door_xform.AddTranslateOp().Set(Gf.Vec3d(*door_offset))
    yaw_rad = math.radians(door_yaw)
    door_xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(math.cos(yaw_rad / 2), Gf.Vec3d(0, 0, math.sin(yaw_rad / 2)))
    )
    print(
        f"Vrata referencirana na {door_path} -> {args_cli.door} "
        f"(offset={door_offset}, yaw={door_yaw}°)"
    )

    # --- Ground plane + physics scene ---
    if not args_cli.skip_ground_plane:
        add_ground_plane(stage)
    add_physics_scene_if_missing(stage)

    stage.GetRootLayer().Save()
    print(f"\nIntegracijska scena spremljena: {args_cli.output}")


if __name__ == "__main__":
    main()
    simulation_app.close()

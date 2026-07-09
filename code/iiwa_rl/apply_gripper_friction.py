"""Dodaje PhysX friction material na collision geometriju prstiju grippera, POSLIJE
URDF->USD konverzije. Trenje nije dio URDF sheme niti convert_urdf.py procesa -
UsdPhysics.MaterialAPI se mora eksplicitno primijeniti na USD stage naknadno.

Pokretanje (headless, ne treba puni Isaac Sim GUI):
  ./isaaclab.sh -p apply_gripper_friction.py output/kmr_iiwa_full.usd

Isto ponovi za URDF/USD vrata (revolute_door.usd, sliding_door.usd) sa
NAME_FILTER = "handle" (ili kako već nazoveš link kvake) - podsjetnik iz
handoff dokumenta: bez frikcije na OBJE strane kontakta, hvat ne drzi
bez obzira na geometriju prstiju.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Primijeni friction material na collision prime po imenu.")
parser.add_argument("usd_path", type=str, help="Putanja do USD filea (mijenja se in-place).")
parser.add_argument("--name-filter", type=str, default="gripper_finger",
                     help="Podstring koji mora sadržavati prim path da bi dobio material (npr. 'gripper_finger' ili 'handle').")
parser.add_argument("--static-friction", type=float, default=1.2)
parser.add_argument("--dynamic-friction", type=float, default=1.0)
parser.add_argument("--restitution", type=float, default=0.0)
parser.add_argument("--material-path", type=str, default="/World/PhysicsMaterials/GripperFingerFriction")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics, UsdShade  # noqa: E402


def main():
    stage = Usd.Stage.Open(args_cli.usd_path)
    if stage is None:
        raise ValueError(f"Ne mogu otvoriti USD: {args_cli.usd_path}")

    # obiđi cijeli stage UKLJUČUJUĆI instance proxije (make_instanceable=False i dalje ne
    # sprječava importer da interno dijeli identične mesheve - vidi napomenu na dnu filea).
    # Svaki pogodak rješavamo isto: nađi stvarni backing layer (GetPrimStack) i binduj TAMO -
    # radi jednako za instance proxy, prototype-derived, i "obične" primove, pa nema
    # posebnog slučaja koji bi mogao ponovno pući na authoring-restrikciji.
    matches = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        path_str = prim.GetPath().pathString
        if args_cli.name_filter not in path_str:
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            matches.append((path_str, prim))

    resolved_layers = {}  # (layer_identifier, prim_path) -> True, dedup (sva 4 prsta dijele isti backing layer)
    unresolved = []
    for instance_path, prim in matches:
        source = prim.GetPrimInPrototype() if prim.IsInstanceProxy() else prim
        stack = source.GetPrimStack()
        if not stack:
            unresolved.append(instance_path)
            continue
        spec = stack[0]  # najjača (prva) opinion - stvarni izvor geometrije
        layer_id = spec.layer.realPath or spec.layer.identifier
        key = (layer_id, str(spec.path))
        if key in resolved_layers:
            print(f"   (preskačem, isti backing layer kao prethodni: {instance_path})")
            continue
        sub_stage = Usd.Stage.Open(layer_id)
        sub_prim = sub_stage.GetPrimAtPath(spec.path)
        if not (sub_prim and sub_prim.IsValid()):
            unresolved.append(f"{instance_path}  (layer: {layer_id}  path: {spec.path})")
            continue
        # materijal treba postojati i unutar OVOG layera (cross-layer reference na main
        # stage material ne rezolvira se kad se ovaj layer otvori samostalno)
        sub_mat_prim = sub_stage.DefinePrim(args_cli.material_path, "Material")
        sub_phys_mat = UsdPhysics.MaterialAPI.Apply(sub_mat_prim)
        sub_phys_mat.CreateStaticFrictionAttr().Set(args_cli.static_friction)
        sub_phys_mat.CreateDynamicFrictionAttr().Set(args_cli.dynamic_friction)
        sub_phys_mat.CreateRestitutionAttr().Set(args_cli.restitution)
        sub_material = UsdShade.Material(sub_mat_prim)
        UsdShade.MaterialBindingAPI.Apply(sub_prim).Bind(sub_material, materialPurpose="physics")
        sub_stage.Save()
        resolved_layers[key] = True
        print(f"   bindano u backing layeru: {layer_id}  [{spec.path}]  (pokriva: {instance_path} i sve ostale instance istog mesha)")

    if unresolved:
        print(f"!! {len(unresolved)} prim(a) se nije moglo resolvati na editabilan layer:")
        for p in unresolved:
            print("    ", p)
        print("   Otvori USD u Isaac Sim GUI-u, desni klik na prim -> 'Select Instance' / provjeri")
        print("   Layer stack panel da ručno nađeš u kojem je fileu stvarno definiran.")

    if not matches:
        print(f"!! Nijedan collision prim ne sadrži '{args_cli.name_filter}' u pathu.")
        print("   Otvori stage u Isaac Sim GUI-u (Window > Stage) i provjeri točan naziv prima")
        print("   koji je importer generirao za linkove grippera, pa ponovi s --name-filter.")
    elif resolved_layers:
        print(f"Gotovo — friction material ({args_cli.static_friction}/{args_cli.dynamic_friction}) "
              f"primijenjen u {len(resolved_layers)} backing layer(a).")


if __name__ == "__main__":
    main()
    simulation_app.close()

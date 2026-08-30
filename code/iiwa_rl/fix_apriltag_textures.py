"""Dodaje diffuse_texture na AprilTag materijale, POSLIJE URDF->USD konverzije.

convert_urdf.py dosljedno ne poveže URDF-ov <texture filename="..."/> na
diffuse_texture shader input (materijal/binding/geometrija se stvore ispravno,
veza do teksture nikad ne nastane) - poznato ograničenje konvertera, ne nešto
rješivo u samom URDF-u. Ovo je post-processing korak koji se mora ponoviti
nakon SVAKE ponovne konverzije (fresh convert_urdf.py briše ovaj fix).

Radi za bilo koji URDF s istom AprilTag anchor konvencijom (root prim ime se
auto-detektira iz defaultPrim-a, pa isti poziv radi i za revolute_door_base.usd
i za sliding_door_base.usd bez izmjena).

Pokretanje (headless):
  ./isaaclab.sh -p fix_apriltag_textures.py \
      output/revolute_door_base.usd

  ./isaaclab.sh -p fix_apriltag_textures.py \
      output/sliding_door_base.usd

Default mapping (3 poznata AprilTag materijala, ista imena u oba URDF-a) se
može override-ati --material zastavicom ako dodaš još tagova ili promijeniš
nazive u URDF-u:

  ./isaaclab.sh -p fix_apriltag_textures.py output/revolute_door_base.usd \
      --material door_tag_material=custom_door.png \
      --material handle_tag_a_material=custom_a.png

NAPOMENA: teksture moraju fizički sjediti u istoj mapi kao usd_path (ili u
--texture-dir ako je zadan) - relativna referenca, isto ograničenje kao i
sam URDF <texture> tag.
"""

import argparse

from isaaclab.app import AppLauncher

DEFAULT_MATERIALS = {
    "door_tag_material": "door_tag_id0_tag36h11.png",
    "handle_tag_a_material": "handle_tag_a_id0_tag16h5.png",
    "handle_tag_b_material": "handle_tag_b_id1_tag16h5.png",
}

parser = argparse.ArgumentParser(description="Popravi diffuse_texture na AprilTag materijalima nakon URDF->USD konverzije.")
parser.add_argument("usd_path", type=str, help="Putanja do _base.usd (mijenja se in-place).")
parser.add_argument("--texture-dir", type=str, default=None,
                     help="Mapa gdje su PNG teksture. Default: ista mapa kao usd_path.")
parser.add_argument("--material", action="append", default=None, metavar="URDF_MATERIAL_NAME=TEXTURE.png",
                     help="Par 'ime_materijala_iz_urdfa=ime_teksture.png'. Može se ponoviti za više parova. "
                          "Default (bez ove zastavice): sva 3 poznata AprilTag materijala "
                          "(door_tag_material, handle_tag_a_material, handle_tag_b_material).")
parser.add_argument("--roughness", type=float, default=0.9,
                     help="reflection_roughness_constant - papir/naljepnica je mat, ne sjajno.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402

from pxr import Sdf, Usd, UsdShade  # noqa: E402


def main():
    stage = Usd.Stage.Open(args_cli.usd_path)
    if stage is None:
        raise ValueError(f"Ne mogu otvoriti USD: {args_cli.usd_path}")

    root_name = stage.GetDefaultPrim().GetName()
    texture_dir = args_cli.texture_dir or os.path.dirname(os.path.abspath(args_cli.usd_path))

    if args_cli.material:
        materials = dict(pair.split("=", 1) for pair in args_cli.material)
    else:
        materials = DEFAULT_MATERIALS

    fixed = []
    missing_prim = []
    missing_texture = []

    for urdf_material_name, texfile in materials.items():
        shader_path = f"/{root_name}/Looks/material_{urdf_material_name}/Shader"
        prim = stage.GetPrimAtPath(shader_path)
        if not prim.IsValid():
            missing_prim.append(shader_path)
            continue

        texture_path = os.path.join(texture_dir, texfile)
        if not os.path.isfile(texture_path):
            missing_texture.append(texture_path)
            continue

        shader = UsdShade.Shader(prim)
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(f"./{texfile}"))
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(args_cli.roughness)
        fixed.append(f"{shader_path} -> {texfile}")

    if fixed:
        stage.GetRootLayer().Save()
        print(f"Popravljeno {len(fixed)} materijal(a) u '{root_name}':")
        for line in fixed:
            print("   ", line)

    if missing_prim:
        print(f"!! {len(missing_prim)} materijal(a) ne postoji na očekivanoj putanji (root prim = '{root_name}'):")
        for p in missing_prim:
            print("    ", p)
        print("   Provjeri imena materijala u URDF-u, ili proslijedi ispravna preko --material.")

    if missing_texture:
        print(f"!! {len(missing_texture)} teksture nije nađeno na disku:")
        for p in missing_texture:
            print("    ", p)
        print("   Provjeri --texture-dir, ili kopiraj PNG-ove pored usd_path.")

    if not fixed and not missing_prim and not missing_texture:
        print("!! Nista za napraviti - materials rjecnik je prazan?")


if __name__ == "__main__":
    main()
    simulation_app.close()

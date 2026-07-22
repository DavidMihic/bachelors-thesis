"""
add_camera_ros_graph.py - Dodaje Camera prim + ROS2 Action Graph (RGB + CameraInfo
publish) na kmr_iiwa_full_base.usd, POSLIJE URDF->USD konverzije.

Zasto ovo treba postojati: convert_urdf.py ne stvara ni pravi Camera prim (URDF
sensor tagovi se ne parsiraju za nas format/verziju), ni ROS2 OmniGraph (nijedan
importer to ne radi automatski - svaki sluzbeni tutorial to opisuje kao odvojen,
rucni korak). Bez ove skripte, svaka ponovna konverzija robota (npr. nakon
promjene krutosti zglobova ruke) zahtijeva RUCNO ponovno postavljanje kamere
u Isaac Sim GUI-u.

Sve vrijednosti nize su izvucene direktno iz stvarno-postavljenog i testiranog
rezultata (Camera transform, Action Graph node tipovi/konekcije/parametri) -
ne nagadjanje. Vidi ARHITEKTURNA NAMJERA komentare uz svaku grupu vrijednosti.

Pokretanje (headless), nakon svake convert_urdf.py konverzije robota:
    ./isaaclab.sh -p add_camera_ros_graph.py assets/configuration/kmr_iiwa_full_base.usd

Idempotentno - siguran za visestruko pokretanje na istom fajlu (brise i
ponovno gradi Camera prim/Graph ako vec postoje, umjesto da duplicira).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dodaje Camera prim + ROS2 Action Graph na robota nakon URDF->USD konverzije.")
parser.add_argument("usd_path", type=str, help="Putanja do _base.usd (mijenja se in-place).")
parser.add_argument("--camera-parent", type=str, default="camera_color_optical_frame",
                     help="Ime linka na koji se Camera prim vjesa (mora vec postojati iz URDF-a).")
parser.add_argument("--rgb-topic", type=str, default="/camera/color/image_raw")
parser.add_argument("--info-topic", type=str, default="camera/color/camera_info")
parser.add_argument("--frame-id", type=str, default="camera_color_optical_frame")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.graph.core as og  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402

# Headless AppLauncher ne ucitava ROS2 bridge ekstenziju automatski (GUI aplikacija
# je ima ukljucenu po defaultu, otud da ovo nije bio problem dok se radilo rucno).
# Bez ovoga, stvaranje ROS2Context/ROS2CameraHelper/ROS2CameraInfoHelper node-ova
# puca s "Could not create node using unrecognized type".
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

CAMERA_NAME = "Camera"


def find_prim_by_name(stage: Usd.Stage, name: str):
    """Link moze zavrsiti pod razlicitim root prim imenima ovisno o defaultPrimu -
    trazimo po imenu, ne po punoj putanji."""
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return prim
    return None


def main():
    usd_context = omni.usd.get_context()
    success = usd_context.open_stage(args_cli.usd_path)
    if not success:
        raise ValueError(f"Ne mogu otvoriti USD kroz omni.usd context: {args_cli.usd_path}")
    # stage load moze biti async - par update() poziva da se sigurno zavrsi prije
    # nego pocnemo editirati (isti pattern kao u sluzbenim Isaac Sim primjerima)
    for _ in range(5):
        simulation_app.update()
    stage = usd_context.get_stage()
    if stage is None:
        raise ValueError("omni.usd context nema aktivan stage nakon open_stage - neocekivano.")

    parent = find_prim_by_name(stage, args_cli.camera_parent)
    if parent is None:
        raise ValueError(f"Link '{args_cli.camera_parent}' ne postoji na stageu - provjeri ime/URDF.")

    # Graph MORA biti unutar defaultPrim podstabla (npr. /kmr_iiwa/Graph/...), ne
    # kao sestrinski root-level prim (/Graph/...) - inace ga reference/payload arc
    # (kojim se ovaj _base.usd uvlaci u wrapper) nece povuci u finalnu scenu,
    # jer reference/payload povlaci SAMO podstablo defaultPrim-a.
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise ValueError("Stage nema postavljen defaultPrim - ne mogu sigurno smjestiti Graph.")
    graph_path = default_prim.GetPath().AppendPath("Graph/ROS_Camera")

    camera_path = parent.GetPath().AppendChild(CAMERA_NAME)

    # ---- Camera prim ----
    # D435 RGB senzor: clipping/focal/aperture izmjereno i potvrdjeno rucnim
    # postavljanjem (vidi raniju raspravu o H-FOV=69.4/V-FOV=42.5 - focalLength
    # 15.13mm + verticalAperture 11.79mm daju ispravan omjer; horizontalAperture
    # NAMJERNO ne postavljamo, ostaje na USD defaultu 20.955mm, isto kao u
    # rucno potvrdjenoj verziji).
    # translate (0,-0.0125,0.0098) = izmjeren offset od tripod-rupe referentne
    # tocke do stvarnog opticnog centra leće (vidi D435 mesh analizu).
    # orient 180 stupnjeva oko X = USD kamera default gleda -Z, nasa optical
    # frame konvencija je Z=naprijed - flip da se poklope.
    if stage.GetPrimAtPath(camera_path).IsValid():
        print(f"Camera prim vec postoji na {camera_path}, brisem i gradim iznova")
        stage.RemovePrim(camera_path)

    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
    camera.CreateFocalLengthAttr(15.13)
    camera.CreateFocusDistanceAttr(400.0)
    camera.CreateVerticalApertureAttr(11.79)

    xformable = UsdGeom.Xformable(camera.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(0.0, -0.0125, 0.0098))
    xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(6.123233995736766e-17, Gf.Vec3d(1, 0, 0))
    )
    xformable.AddScaleOp().Set(Gf.Vec3f(1, 1, 1))
    print(f"Camera prim kreiran na {camera_path}")

    # ---- ROS2 Action Graph ----
    # 6 node-ova, tocne konekcije/parametri izvuceni iz potvrdjeno-ispravnog
    # rezultata: OnPlaybackTick -> RunOnce -> RenderProduct -> {RGBPublish, CameraInfoPublish},
    # Context.outputs:context -> oba publisher node-a.
    if stage.GetPrimAtPath(str(graph_path)).IsValid():
        print(f"Graph vec postoji na {graph_path}, brisem i gradim iznova")
        stage.RemovePrim(str(graph_path))
        stage.GetRootLayer().Save()

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": str(graph_path), "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("RunOnce", "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("RenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("RGBPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("CameraInfoPublish", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "RunOnce.inputs:execIn"),
                ("RunOnce.outputs:step", "RenderProduct.inputs:execIn"),
                ("RenderProduct.outputs:execOut", "RGBPublish.inputs:execIn"),
                ("RenderProduct.outputs:execOut", "CameraInfoPublish.inputs:execIn"),
                ("RenderProduct.outputs:renderProductPath", "RGBPublish.inputs:renderProductPath"),
                ("RenderProduct.outputs:renderProductPath", "CameraInfoPublish.inputs:renderProductPath"),
                ("Context.outputs:context", "RGBPublish.inputs:context"),
                ("Context.outputs:context", "CameraInfoPublish.inputs:context"),
            ],
            keys.SET_VALUES: [
                ("RenderProduct.inputs:cameraPrim", [str(camera_path)]),
                ("RenderProduct.inputs:enabled", True),
                ("RenderProduct.inputs:width", args_cli.width),
                ("RenderProduct.inputs:height", args_cli.height),
                ("RGBPublish.inputs:frameId", args_cli.frame_id),
                ("RGBPublish.inputs:nodeNamespace", ""),
                ("RGBPublish.inputs:topicName", args_cli.rgb_topic),
                ("RGBPublish.inputs:type", "rgb"),
                ("CameraInfoPublish.inputs:frameId", args_cli.frame_id),
                ("CameraInfoPublish.inputs:nodeNamespace", ""),
                ("CameraInfoPublish.inputs:topicName", args_cli.info_topic),
            ],
        },
    )
    print(f"ROS2 Action Graph kreiran na {graph_path}")

    stage.GetRootLayer().Save()
    print(f"\nSpremljeno u {args_cli.usd_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()

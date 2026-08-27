"""add_base_joints.py - ubaci tri fiktivna zgloba za pokretnu bazu.

ZASTO: OSC racuna jakobijan i inercijsku matricu uz pretpostavku FIKSNOG
korijena artikulacije. S plutajucim korijenom (fix_root_link=False)
kompenzacija gravitacije ispadne kriva i ruka propada - provjereno
empirijski. Pomicanje baze zato ne smije ici kroz plutajuci korijen.

RJESENJE: korijen ostaje fiksan, ali se ubaci lanac

    world -> base_x_joint -> base_x_link
          -> base_y_joint -> base_y_link
          -> base_theta_joint -> base_link -> (ostatak robota nepromijenjen)

Baza se time giba kao dio artikulacije, kroz zglobove s pravim pogonom, pa
solver sam razrjesava sile umjesto da mu se stanje namece teleportacijom
korijena. To je i razlog zasto ovo nije isto sto i write_root_pose_to_sim:
upis poze je teleportacija koja resetira unutarnje stanje solvera dok
zglobovi zadrze kutove, pa se lanac raspadne.

Prema van sucelje ostaje isto: zakon i dalje racuna (vx, vy, omega), samo se
predaje kao naredba trima zglobovima umjesto kao poza. To je i dalje doslovno
geometry_msgs/Twist na /cmd_vel.

Skripta NE prepisuje original - cita kmr_iiwa_full.urdf i pise novi fajl.
Time se izbjegava odrzavanje dvije kopije istog robota u sinkronu.

Pokretanje (ne treba Isaac, cisti Python):
    python3 code/iiwa_rl/add_base_joints.py \
        code/ros2_ws/src/kmr_iiwa_description/urdf/kmr_iiwa_full.urdf \
        code/ros2_ws/src/kmr_iiwa_description/urdf/kmr_iiwa_full_rl.urdf
"""

import argparse
import xml.etree.ElementTree as ET

# Doseg fiktivnih zglobova. Velikodusno - ogranicenje gibanja baze dolazi iz
# admitancijskog zakona (max_linear_speed), ne iz limita zgloba.
TRANSLATION_LIMIT_M = 10.0
ROTATION_LIMIT_RAD = 6.2832

# Fiktivni linkovi trebaju masu i inerciju jer Isaac ne prihvaca pomicne
# linkove bez njih. Vrijednosti su male ali NE zanemarive: uz platformu od
# 390 kg omjer masa ostaje oko 390:1, sto solver podnosi. Omjeri reda
# 10000:1 unutar iste artikulacije su ono sto izaziva jitter.
DUMMY_MASS_KG = 1.0
DUMMY_INERTIA = 0.01


def dummy_link(name: str) -> ET.Element:
    link = ET.Element("link", {"name": name})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": str(DUMMY_MASS_KG)})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": str(DUMMY_INERTIA),
            "iyy": str(DUMMY_INERTIA),
            "izz": str(DUMMY_INERTIA),
            "ixy": "0",
            "ixz": "0",
            "iyz": "0",
        },
    )
    return link


def base_joint(
    name: str, parent: str, child: str, joint_type: str, axis: str, limit: float
) -> ET.Element:
    joint = ET.Element("joint", {"name": name, "type": joint_type})
    ET.SubElement(joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    ET.SubElement(joint, "axis", {"xyz": axis})
    ET.SubElement(
        joint,
        "limit",
        {
            "lower": str(-limit),
            "upper": str(limit),
            # Platforma je ~390 kg; effort mora biti dovoljan da je pokrene.
            "effort": "5000",
            "velocity": "2.0",
        },
    )
    # Bez prigusenja fiktivni zglobovi oscilaraju pod reakcijom ruke.
    ET.SubElement(joint, "dynamics", {"damping": "50.0", "friction": "0.0"})
    return joint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_urdf")
    parser.add_argument("output_urdf")
    parser.add_argument(
        "--root-link",
        default="base_link",
        help="Link na koji se lanac spaja (korijen izvornog URDF-a).",
    )
    args = parser.parse_args()

    tree = ET.parse(args.input_urdf)
    robot = tree.getroot()

    existing = {link.get("name") for link in robot.findall("link")}
    if args.root_link not in existing:
        raise SystemExit(f"Link '{args.root_link}' ne postoji u {args.input_urdf}")
    if "world" in existing:
        raise SystemExit("URDF vec ima 'world' link - zglobovi su vjerojatno vec ubaceni")

    robot.set("name", robot.get("name", "kmr_iiwa") + "_rl")

    # world nema inercijalni blok - time ga Isaac tretira kao fiksnu bazu, sto
    # je upravo ono sto OSC-u treba.
    additions = [
        ET.Element("link", {"name": "world"}),
        dummy_link("base_x_link"),
        dummy_link("base_y_link"),
        base_joint(
            "base_x_joint", "world", "base_x_link", "prismatic", "1 0 0",
            TRANSLATION_LIMIT_M,
        ),
        base_joint(
            "base_y_joint", "base_x_link", "base_y_link", "prismatic", "0 1 0",
            TRANSLATION_LIMIT_M,
        ),
        base_joint(
            "base_theta_joint", "base_y_link", args.root_link, "revolute", "0 0 1",
            ROTATION_LIMIT_RAD,
        ),
    ]

    # Na pocetak, da je lanac citljiv na vrhu fajla.
    for i, element in enumerate(additions):
        robot.insert(i, element)

    ET.indent(tree, space="  ")
    tree.write(args.output_urdf, encoding="utf-8", xml_declaration=True)
    print(f"Zapisano: {args.output_urdf}")
    print("Novi zglobovi: base_x_joint, base_y_joint, base_theta_joint")


if __name__ == "__main__":
    main()

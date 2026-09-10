"""wall_cfg.py - statični zidovi oko vrata.

Zid NIJE dio URDF-a vrata: vrata se pri resetu postavljaju na nasumičnu pozu i
rotiraju za delta, a zid mora stajati u fiksnom odnosu prema OKVIRU vrata, ne
uz njih kao dio istog tijela. Zato je zaseban statični asset u sceni, kao pod
i svjetlo, i postavlja se iz iste poze na koju i vrata.

Dvije konfiguracije:

  ZAKRETNA - zid lijevo i desno od otvora, u ravnini zatvorenih vrata. Otvor
             izmedju njih je sirok kao krilo plus mali zazor. Krilo se okrece
             u prostor ISPRED zida (prema robotu), pa mu bocni zidovi ne
             smetaju - a baza koja bi se htjela provozati kroz otvor sad
             udara u dovratnik/zid umjesto da rasklopi krilo.

  KLIZNA   - jedan zid, pomaknut IZA ravnine vrata za clearance, da krilo koje
             klizi u stranu ne zapne za njega. Zid je s one strane u koju se
             vrata NE otvaraju.

Geometrija je u LOKALNOM okviru vrata (isti okvir u kojem su HANDLE_LOCAL_*),
pa se transformira zajedno s pozom vrata pri resetu - vidi reset u door_mdp.

Dimenzije (parametri dolje):
  debljina zida     0.10 m
  visina zida       2.10 m (nesto vise od vrata da se ne moze preci preko)
  sirina krila      0.85 m (iz URDF-a)
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

# --- dimenzije ---
WALL_THICKNESS_M = 0.10
WALL_HEIGHT_M = 2.10
WALL_WIDTH_M = 1.20  # sirina jednog segmenta zida uz otvor

LEAF_WIDTH_M = 0.85
SIDE_GAP_M = 0.05  # zazor izmedju ruba krila i zida kod zakretnih
BACK_OFFSET_M = 0.15  # koliko je zid IZA ravnine vrata kod kliznih

# Lokalne poze segmenata zida u okviru vrata (prije nego reset primijeni
# stvarnu pozu). Konvencija okvira vrata: sarka/ishodiste u (0,0,0), krilo se
# pruza duz +Y, normala vrata je +X, visina +Z.

# ZAKRETNA: dva zida u ravnini X=0, lijevo (Y<0, iza sarke) i desno (Y iza
# slobodnog ruba krila). Otvor je izmedju Y=0 i Y=LEAF_WIDTH.
REVOLUTE_WALL_LOCAL = [
    # lijevo od sarke
    (0.0, -(WALL_WIDTH_M / 2.0 + SIDE_GAP_M), WALL_HEIGHT_M / 2.0),
    # desno od slobodnog ruba
    (
        0.0,
        LEAF_WIDTH_M + SIDE_GAP_M + WALL_WIDTH_M / 2.0,
        WALL_HEIGHT_M / 2.0,
    ),
]
REVOLUTE_WALL_SIZE = (WALL_THICKNESS_M, WALL_WIDTH_M, WALL_HEIGHT_M)

# KLIZNA: jedan zid iza ravnine vrata (X = -BACK_OFFSET), pokriva sirinu
# otvora. Krilo klizi u stranu ispred zida, pa ne zapinje.
SLIDING_WALL_LOCAL = [
    (-BACK_OFFSET_M, LEAF_WIDTH_M / 2.0, WALL_HEIGHT_M / 2.0),
]
SLIDING_WALL_SIZE = (WALL_THICKNESS_M, LEAF_WIDTH_M + 2 * WALL_WIDTH_M, WALL_HEIGHT_M)


def wall_asset_cfg(prim_path: str, size: tuple[float, float, float]) -> AssetBaseCfg:
    """Statični zid kao krut kolider bez artikulacije. Poza se postavlja u
    resetu, ovdje je samo geometrija."""
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,  # nepomican, ali sudjeluje u koliziji
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.6, 0.62)
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0)),
    )

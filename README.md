# Razvoj i evaluacija metode upravljanja mobilnim manipulatorom u interakciji s okolinom.

U ovom zadatku razmatra se problem autonomne interakcije mobilnog manipulatora s okolinom kroz zadatke koji uključuju uspostavu i održavanje kontakta s objektima te izvođenje koordiniranih manipulacijskih radnji. Naglasak je na integraciji percepcije, procjene stanja i upravljanja u jedinstveni sustav sposoban za izvršavanje višefaznih zadataka u nepoznatim ili djelomično poznatim uvjetima. Cilj rada je razviti i eksperimentalno evaluirati metodologiju upravljanja koja omogućuje robotu donošenje odluka i generiranje gibanja na temelju više senzorskih podataka, uzimajući u obzir mehanička svojstva okoline i interakcijske sile.

U radu je potrebno:
- istražiti postojeće pristupe upravljanju mobilnim manipulatorima u zadacima koji uključuju fizičku interakciju s objektima,
- razviti modul percepcije za detekciju relevantnih objekata te procjenu njihove pozicije i orijentacije korištenjem dostupnih senzora,
- implementirati postupak procjene svojstava okoline (npr. krutost, ograničenja gibanja) na temelju mjerenja sile i momenta tijekom interakcije, 
- modelirati i implementirati dva različita pristupa upravljanju sustavom (klasični pristup temeljen na modelu i pristup temeljen na učenju),
- definirati i implementirati odgovarajuću reprezentaciju stanja, akcije i kriterija uspješnosti za odabrani pristup učenja,
- analizirati i usporediti performanse implementiranih metoda s obzirom na učinkovitost i robusnost sustava.


## Setup
USD file se generira iz URDF-a pomoću Isaac Sim alata.

```
conda activate isaacsim
cd ~/IsaacLab
./isaaclab.sh -p scripts/tools/convert_urdf.py 
<repo>/robot/urdf/iiwa7.urdf <repo>/assets/iiwa7.usd 
--merge-joints --joint-stiffness 0 --joint-damping 0
```

## Structure
- `code/`
    - `iiwa7/` — RL scripts (Isaac Lab tasks, configs)
    - `ros2_ws/`
        - `kmr_iiwa_description`
            - `urdf/` — cleaned URDF
            - `xacro/` — original xacro sources
            - `meshes/` — STL meshes
        - `kmr_iiwa_sim_bridge` — ROS2 bridge for IsaacSim
- `assets/` — generated USD
- `latex/` — LaTeX source
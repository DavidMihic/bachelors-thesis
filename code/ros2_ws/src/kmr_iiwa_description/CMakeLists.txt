cmake_minimum_required(VERSION 3.8)
project(kmr_iiwa_description)

find_package(ament_cmake REQUIRED)

# Kopira cijele foldere (urdf, meshes) u install/share/kmr_iiwa_description/
install(
  DIRECTORY urdf meshes
  DESTINATION share/${PROJECT_NAME}
)

# Ako kasnije dodas launch/ folder (npr. za robot_state_publisher + RViz),
# otkomentiraj:
# install(
#   DIRECTORY launch
#   DESTINATION share/${PROJECT_NAME}
# )

ament_package()
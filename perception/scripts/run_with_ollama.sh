# Host paths to mount into the container; set these before running.
: "${QUAD_WS_DIR:?Set QUAD_WS_DIR to the host path of your quad_ws}"
: "${JSONS_DIR:?Set JSONS_DIR to the host path for scene graph JSON files}"
: "${JETSON_CONTAINERS_DATA:?Set JETSON_CONTAINERS_DATA to the host path of jetson-containers/data}"

docker run --runtime nvidia -it --rm --network host --volume /tmp/argus_socket:/tmp/argus_socket --volume /etc/enctune.conf:/etc/enctune.conf --volume /etc/nv_tegra_release:/etc/nv_tegra_release --volume /var/run/dbus:/var/run/dbus --volume /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket --volume /var/run/docker.sock:/var/run/docker.sock --volume ${JETSON_CONTAINERS_DATA}:/data --volume ${QUAD_WS_DIR}:/workspace/quad_ws --volume ${JSONS_DIR}:/workspace/jsons --device /dev/snd --device /dev/bus/usb --device /dev/i2c-0 --device /dev/i2c-1 --device /dev/i2c-2 --device /dev/i2c-3 --device /dev/i2c-4 --device /dev/i2c-5 --device /dev/i2c-6 --device /dev/i2c-7 --device /dev/i2c-8 --device /dev/i2c-9 scene_graph

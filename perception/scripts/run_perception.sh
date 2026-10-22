# Host paths to mount into the container; set these before running.
: "${QUAD_WS_DIR:?Set QUAD_WS_DIR to the host path of your quad_ws}"
: "${JSONS_DIR:?Set JSONS_DIR to the host path for scene graph JSON files}"
: "${BAGFILE_DIR:?Set BAGFILE_DIR to the host path of your rosbag2 directory}"

docker run -it \
  --runtime=nvidia \
  --privileged \
  --rm \
  --shm-size=10g \
  --net=host \
  \
  --mount type=bind,source=$(pwd)/ws/src,target=/workspace/ws/src,consistency=delegated \
  \
  --mount type=bind,source=$(pwd)/Dockerfile,target=/workspace/Dockerfile,consistency=cached \
  --mount type=bind,source=$(pwd)/.vscode,target=/workspace/.vscode,consistency=cached \
  --mount type=bind,source=$(pwd)/.devcontainer,target=/workspace/.devcontainer,consistency=cached \
  \
  --mount type=bind,source=${QUAD_WS_DIR},target=/workspace/quad_ws,consistency=cached \
  --mount type=bind,source=${BAGFILE_DIR},target=/workspace/bagfile,consistency=cached \
  --mount type=bind,source=${JSONS_DIR},target=/workspace/jsons,consistency=cached \
  \
  --workdir /workspace \
  go2-llm-perception \
  /bin/bash


# Perception

The perception pipeline of OrionNav has two critical components: Open-Vocabulary Segmentation and Scene Graph Generation. Before running these modules, build the relevant docker containers using the following commands

```
docker build -t go2-llm-perception -f perception/deploy/Dockerfile_seg .
docker build -t scene_graph -f  perception/deploy/Dockerfile_dsg .  # for scene graph generation
```

To build the necessary ros workspaces, run the following:
```
cd perception/seg_ws
catkin build
cd ../dsg_ws
catkin build
```

Finally, before running the individual modules, run `./scripts/source_ros_ws.sh` to source all relevant ros workspaces

### Open-Vocabulary Segmentation

Run `./scripts/run_perception.sh' to enter the docker container, and then run the following to start the segmentation
```
ros2 launch mask_to_bbox_cpp perception.launch.py save_to_json:=true 
```

### Scene Graph Generation

To run the scene graph generation code, enter the docker container using `./scripts/run_with_ollama.sh`, and then run the following command.
```
ros2 launch scene_graph gen_sg_launch.py
```

Make sure to run the segmentation first and then the scene graph generation.

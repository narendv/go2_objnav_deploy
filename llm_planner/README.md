# orionnav-llm-planner

This folder contains the LLM Planner component of the repository. It must be run directly on the robot.

The Conda environment for running the LLM Planner is provided in `environment.yaml`.

### Setup and Execution

1. Navigate to the `llm_planner` directory:
    ```bash
    cd <llm_planner_path>
    ```

2. Activate the Conda environment:
    ```bash
    conda activate llm_env
    ```

3. Run the LLM Planner script:
    ```bash
    python src/main.py --jetson-ip <your_jetson_ip> --jetson-username <your_jetson_username> --jetson-password <your_jetson_password> --map-path <map_path_in_jetson> --openai-key <openai_key>
    ```

Make sure to replace the placeholder values (`<your_jetson_ip>`, `<your_jetson_username>`, etc.) with your actual information.

from __future__ import annotations
from typing import Dict
import pathlib

try:
    import zoo
except ImportError:
    class ZooStub(object):
        def __init__(self):
            self.SERVICE_SUCCEEDED = 3
            self.SERVICE_FAILED = 4

        def update_status(self, conf, progress):
            print(f"Status {progress}")

        def _(self, message):
            print(f"invoked _ with {message}")

    zoo = ZooStub()

import os
import sys
import traceback
import yaml
import json
import boto3  # noqa
import botocore
from loguru import logger
from urllib.parse import urlparse
from botocore.exceptions import ClientError
from botocore.client import Config
from pystac import read_file
from pystac.stac_io import DefaultStacIO, StacIO
from pystac.item_collection import ItemCollection
from zoo_calrissian_runner import ExecutionHandler, ZooCalrissianRunner
from pycalrissian.context import CalrissianContext

logger.remove()
logger.add(sys.stderr, level="INFO")


class SimpleExecutionHandler(ExecutionHandler):
    def __init__(self, conf, resources):
        super().__init__()
        self.conf = conf
        self.results = None
        self._resources = resources  # Store pod resources

    def get_pod_resources(self):
        logger.info("get_pod_resources: " + str(self._resources))
        return {
            "requests": {"cpu": "1", "memory": "2Gi", "nvidia.com/gpu": "1"},
            "limits": {"cpu": "1", "memory": "2Gi", "nvidia.com/gpu": "1"}
        }

    def pre_execution_hook(self):
        logger.info("Pre execution hook")
        input_request = self.conf['request']['jrequest']
        service_name = json.loads(input_request)['inputs']['thematic_service_name']
        logger.info(f"Thematic service name: {service_name}")

        stageout_yaml = yaml.safe_load(open("/assets/stageout.yaml", "rb"))
        logger.info(f"Stageout: {stageout_yaml}")

        self.stageout_file_path = f"/{self.conf['main']['tmpPath']}/stageout{self.conf['lenv']['usid']}.yaml"
        with open(self.stageout_file_path, "w") as stageout_file:
            yaml.dump(stageout_yaml, stageout_file)
        os.environ["WRAPPER_STAGE_OUT"] = self.stageout_file_path

    def post_execution_hook(self, log, output, usage_report, tool_logs):
        os.environ.pop("HTTP_PROXY", None)
        logger.info("Post execution hook")

    def get_pod_env_vars(self):
        logger.info("get_pod_env_vars")
        return {
            "ANOTHER_VAR": self.conf['pod_env_vars']['ANOTHER_VAR'],
            "S3_BUCKET_NAME": self.conf['pod_env_vars']['S3_BUCKET_ADDRESS'],
            "AWS_ACCESS_KEY_ID": self.conf['pod_env_vars']['BUCKET_1_AK'],
            "AWS_SECRET_ACCESS_KEY": self.conf['pod_env_vars']['BUCKET_1_AS'],
            "AWS_DEFAULT_REGION": "eu-central-1",
            "PROCESS_ID": self.conf["lenv"]["usid"]
        }

    def get_pod_node_selector(self):
        logger.info("get_pod_node_selector")
        node_selector = {}
        gpu_requested = (
            "nvidia.com/gpu" in self._resources.get("requests", {}) and
            int(self._resources["requests"]["nvidia.com/gpu"]) > 0
        )
        if gpu_requested:
            # Use the current GPU label
            node_selector["accelerator"] = "nvidia"
            logger.info("NodeSelector: Targeting GPU nodes with accelerator=nvidia!")
        else:
            # Optionally target CPU nodegroup
            node_selector["nodegroup"] = "cpu"
            logger.info("NodeSelector: Targeting CPU nodegroup!")
        return node_selector

    def get_additional_parameters(self):
        logger.info("get_additional_parameters")
        additional_parameters = self.conf.get("additional_parameters", {})
        additional_parameters["sub_path"] = self.conf["lenv"]["usid"]
        logger.info(f"additional_parameters: {list(additional_parameters.keys())}")
        logger.info(json.dumps(self.conf))
        return additional_parameters

    def handle_outputs(self, log, output, usage_report, tool_logs):
        try:
            logger.info("handle_outputs")
            logger.info(tool_logs)
            logger.info(output)
            logger.info(log)
            logger.info(usage_report)
        except Exception as e:
            logger.error("ERROR in handle_outputs...")
            logger.error(traceback.format_exc())
            raise e

    def get_secrets(self):
        return {}

def has_gpu_hint(hints):
    if isinstance(hints, dict):
        hints = [hints]
    for h in hints:
        if not isinstance(h, dict):
            continue
        # Check for several GPU hint patterns
        if h.get("class") == "cwltool:CUDARequirement":
            return True
        if h.get("gpuMin", 0) > 0:
            return True
        if h.get("nvidia.com/gpu", 0) > 0:
            return True
    return False
def has_gpu_min_hint(cwl: dict) -> bool:
    """
    Returns True if any 'hints' entry in the given CWL object or its $graph
    contains a top-level gpuMin: >0. Supports both dict and list hints.
    """
    def gpu_min_in_hints(hints):
        if isinstance(hints, dict):
            # Flat dict as hints
            return hints.get("gpuMin", 0) > 0
        if isinstance(hints, list):
            # List of dicts as hints (common pattern)
            for hint in hints:
                if isinstance(hint, dict) and hint.get("gpuMin", 0) > 0:
                    return True
        return False

    # Check top-level hints
    if gpu_min_in_hints(cwl.get("hints", {})):
        return True

    # Check in $graph if present (for packed workflows)
    for entry in cwl.get("$graph", []):
        if gpu_min_in_hints(entry.get("hints", {})):
            return True

    return False


def prepare_resources_from_cwl(cwl: dict) -> dict:
    use_gpu = has_gpu_min_hint(cwl)

    def has_cuda_requirement(hints):
        if isinstance(hints, dict):
            hints = [hints]
        return any(h.get("class") == "cwltool:CUDARequirement" for h in hints if isinstance(h, dict))

    graph_objects = {entry.get("id", "").lstrip("#"): entry for entry in cwl.get("$graph", [])}

    def check_for_cuda_in_steps(workflow):
        for step in workflow.get("steps", []):
            run_ref = step.get("run")
            if isinstance(run_ref, dict):
                if has_cuda_requirement(run_ref.get("hints", [])):
                    return True
            elif isinstance(run_ref, str):
                run_obj = graph_objects.get(run_ref.lstrip("#"))
                if run_obj and has_cuda_requirement(run_obj.get("hints", [])):
                    return True
        return False

    if cwl.get("class") == "Workflow":
        if check_for_cuda_in_steps(cwl):
            use_gpu = True
    elif has_cuda_requirement(cwl.get("hints", [])):
        use_gpu = True
    elif "$graph" in cwl:
        for entry in cwl["$graph"]:
            if entry.get("class") in {"CommandLineTool", "Workflow"} and has_cuda_requirement(entry.get("hints", [])):
                use_gpu = True
                break
    use_gpu = True

    resources = {
        "requests": {"cpu": "1", "memory": "2Gi"},
        "limits": {"cpu": "1", "memory": "2Gi"},
    }

    if use_gpu:
        resources["requests"]["nvidia.com/gpu"] = "1"
        resources["limits"]["nvidia.com/gpu"] = "1"
        logger.info("USING GPU!")
    else:
        logger.info("NOT USING GPU!")

    return resources

def {{cookiecutter.workflow_id |replace("-", "_")}}(conf, inputs, outputs):  # noqa
    try:
        logger.info(inputs)

        # Load CWL definition
        cwl_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "app-package.cwl")
        with open(cwl_path, "r") as stream:
            cwl = yaml.safe_load(stream)

        finalized_cwl = cwl

        # Detect if GPU is needed and define resources accordingly
        resources = prepare_resources_from_cwl(finalized_cwl)
        logger.info(f"Resources: {resources}")

        # Create a Calrissian context with resource limits
        context = CalrissianContext(
            namespace="zoo-job",
            storage_class=conf.get("main", {}).get("storageClass", "standard"),
            volume_size="10Gi"

        )
        context.default_container_resources = resources

        # Setup execution handler and assign context with resource limits
        execution_handler = SimpleExecutionHandler(conf=conf, resources=resources)
        execution_handler.context = context

        # Create the runner
        runner = ZooCalrissianRunner(
            cwl=finalized_cwl,
            conf=conf,
            inputs=inputs,
            outputs=outputs,
            execution_handler=execution_handler
        )

        # Create and switch to working directory
        working_dir = os.path.join(conf["main"]["tmpPath"], runner.get_namespace_name())
        os.makedirs(working_dir, mode=0o777, exist_ok=True)
        os.chdir(working_dir)

        # Execute the workflow
        exit_status = runner.execute()

        if exit_status == zoo.SERVICE_SUCCEEDED:
            return zoo.SERVICE_SUCCEEDED
        else:
            conf["lenv"]["message"] = zoo._("Execution failed")
            logger.error("Execution failed")
            return zoo.SERVICE_FAILED

    except Exception as e:
        logger.error("ERROR in processing execution template...")
        logger.error("Try to fetch the tool logs if any...")

        try:
            if 'runner' in locals() and hasattr(runner, 'execution'):
                tool_logs = runner.execution.get_tool_logs()
                execution_handler.handle_outputs(None, None, None, tool_logs)
        except Exception as tool_log_err:
            logger.error(f"Fetching tool logs failed! ({str(tool_log_err)})")

        conf["lenv"]["message"] = zoo._(f"Exception during execution...\n{traceback.format_exc()}\n")
        return zoo.SERVICE_FAILED

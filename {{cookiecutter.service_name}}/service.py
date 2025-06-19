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
import boto3  # noqa: F401
import botocore
from loguru import logger
from urllib.parse import urlparse
from botocore.exceptions import ClientError
from botocore.client import Config
from pystac import read_file
from pystac.stac_io import DefaultStacIO, StacIO
from pystac.item_collection import ItemCollection
from zoo_calrissian_runner import ExecutionHandler, ZooCalrissianRunner


logger.remove()
logger.add(sys.stderr, level="INFO")



class SimpleExecutionHandler(ExecutionHandler):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.results = None

    def finalize_cwl(self, cwl_in):
        logger.info("Finalize CWL")
        cwl_in_json = json.loads(cwl_in)



        cwl_out = json.dumps(cwl_in_json)
        return cwl_out

    def pre_execution_hook(self):

        logger.info("Pre execution hook")
        input_request = self.conf['request']['jrequest']
        import json
        service_name = json.loads(input_request)['inputs']['thematic_service_name']
        logger.info(f"Thematic service name: {service_name}")
        
        stageout_yaml = yaml.safe_load(open("/assets/stageout.yaml","rb"))
        
        logger.info(f"Stageout: {stageout_yaml}")
        logger.info("WRAPPER_STAGE_OUT" in os.environ)

        self.stageout_file_path = f"/{self.conf['main']['tmpPath']}/stageout{self.conf['lenv']['usid']}.yaml"
        stageout_file=open(self.stageout_file_path,"w")
        yaml.dump(stageout_yaml,stageout_file)
        stageout_file.close()
        os.environ["WRAPPER_STAGE_OUT"] = self.stageout_file_path

        logger.info("WRAPPER_STAGE_OUT" in os.environ)

    def post_execution_hook(self, log, output, usage_report, tool_logs):

        # unset HTTP proxy or else the S3 client will use it and fail
        os.environ.pop("HTTP_PROXY", None)

        logger.info("Post execution hook")


    def get_pod_env_vars(self):
        # This method is used to set environment variables for the pod
        # spawned by calrissian.

        logger.info("get_pod_env_vars")
        
        env_vars = {
            "ANOTHER_VAR": self.conf['pod_env_vars']['ANOTHER_VAR'],
            "S3_BUCKET_NAME": self.conf['pod_env_vars']['S3_BUCKET_ADDRESS'],
            "AWS_ACCESS_KEY_ID":self.conf['pod_env_vars']['BUCKET_1_AK'],
            "AWS_SECRET_ACCESS_KEY": self.conf['pod_env_vars']['BUCKET_1_AS'],
            "AWS_DEFAULT_REGION": "eu-central-1",
            "PROCESS_ID": self.conf["lenv"]["usid"]
        }
        

        return env_vars

    def get_pod_node_selector(self):
        # This method is used to set node selectors for the pod
        # spawned by calrissian.

        logger.info("get_pod_node_selector")

        node_selector = {}

        return node_selector

    def get_additional_parameters(self):
        # sets the additional parameters for the execution
        # of the wrapped Application Package

        logger.info("get_additional_parameters")
        additional_parameters: Dict[str, str] = {}
        additional_parameters = self.conf.get("additional_parameters", {})
        
        additional_parameters["sub_path"] = self.conf["lenv"]["usid"]

        logger.info(f"additional_parameters: {additional_parameters.keys()}")
        import json
        logger.info(json.dumps(self.conf))
        return additional_parameters

    def handle_outputs(self, log, output, usage_report, tool_logs):
        """
        Handle the output files of the execution.

        :param log: The application log file of the execution.
        :param output: The output file of the execution.
        :param usage_report: The metrics file.
        :param tool_logs: A list of paths to individual workflow step logs.

        """

        try:
            logger.info("handle_outputs")
            logger.info(tool_logs)
            logger.info(output)
            logger.info(log)
            logger.info(usage_report)
            
           # logger.info(f"Set output to {output['s3_catalog_output']}")
           
            # services_logs = [
            #     {
            #         "url": os.path.join(
            #             self.conf["main"]["tmpUrl"],
            #             f"{self.conf['lenv']['Identifier']}-{self.conf['lenv']['usid']}",
            #             os.path.basename(tool_log),
            #         ),
            #         "title": f"Tool log {os.path.basename(tool_log)}",
            #         "rel": "related",
            #     }
            #     for tool_log in tool_logs
            # ]
            # for i in range(len(services_logs)):
            #     okeys = ["url", "title", "rel"]
            #     keys = ["url", "title", "rel"]
            #     if i > 0:
            #         for j in range(len(keys)):
            #             keys[j] = keys[j] + "_" + str(i)
            #     if "service_logs" not in self.conf:
            #         self.conf["service_logs"] = {}
            #     for j in range(len(keys)):
            #         self.conf["service_logs"][keys[j]] = services_logs[i][okeys[j]]

            # self.conf["service_logs"]["length"] = str(len(services_logs))
            # logger.info(f"service_logs: {self.conf['service_logs']}")

        except Exception as e:
            logger.error("ERROR in handle_outputs...")
            logger.error(traceback.format_exc())
            raise (e)

    def get_secrets(self):
        return {}

    def update_workflow_graph(self, workflow_graph):
        workflow_graph["outputs"]["execution_results"] = {
            "type": "Directory",
            "outputSource": ["merge_results/execution_results"]
        }

        workflow_graph["steps"] = {
            "analyse": {
                "run": "#analyse",
                "in": {"spatial_extent": "spatial_extent"},
                "out": ["data_analysis_results"],
            },
            "stageout_data_analysis": {
                "run": "#stageout_data_analysis",
                "in": {"data_analysis_results": "analyse/data_analysis_results"},
                "out": ["stageout_data_analysis_results"],
            },
            "split_tiles": {
                "run": "#split_tiles",
                "in": {
                    "spatial_extent": "spatial_extent",
                    "data_analysis_results": "analyse/data_analysis_results",
                    "stageout_data_analysis_results": "stageout_data_analysis/stageout_data_analysis_results",
                },
                "out": ["split_tiles_results", "tiles"],
            },
            "process": {
                "run": "#process",
                "in": {
                    "spatial_extent": {
                        "source": "split_tiles/tiles",
                        "valueFrom": "$(self.spatial_extent)",
                    }
                },
                "out": ["process_results"],
                "scatter": "spatial_extent",
                "scatterMethod": "flat_crossproduct",
            },
            "merge_results": {
                "run": "#merge_results",
                "in": {
                    "process_results": "process/process_results"
                },
                "out": ["execution_results"],
            },
        }
        return workflow_graph

    def update_process_graph(self, process_graph):
        process_graph["inputs"] = {
            "spatial_extent": {
                "type": "string[]"
            }
        }
        return process_graph

    def add_stageout_data_analysis_graph(self):
        return {
            "class": "CommandLineTool",
            "id": "stageout_data_analysis",
            "baseCommand": "python",
            "arguments": [
                "/app/stageout_data_analysis.py",
                "--data_analysis_results",
                "$(inputs.data_analysis_results)",
            ],
            "requirements": {
                "ResourceRequirement": {
                    "coresMax": 1,
                    "ramMax": 512,
                }
            },
            "hints": {
                "DockerRequirement": {
                    "dockerPull": "pminel/zoo_reference_implementation_v4",
                }
            },
            "inputs": {
                "data_analysis_results": {
                    "type": "Directory"
                }
            },
            "outputs": {
                "stageout_data_analysis_results": {
                    "type": "Directory",
                    "outputBinding": {
                        "glob": "."
                    }
                }
            }
        }

    def add_split_tiles_graph(self):
        return {
            "class": "CommandLineTool",
            "id": "split_tiles",
            "baseCommand": "python",
            "arguments": [
                "/app/split_tiles.py",
                "--spatial_extent",
                "$(inputs.spatial_extent[0])",
                "$(inputs.spatial_extent[1])",
                "$(inputs.spatial_extent[2])",
                "$(inputs.spatial_extent[3])",
                "--data_analysis_results",
                "$(inputs.data_analysis_results)",
            ],
            "requirements": {
                "ResourceRequirement": {
                    "coresMax": 1,
                    "ramMax": 512,
                }
            },
            "hints": {
                "DockerRequirement": {
                    "dockerPull": "pminel/zoo_reference_implementation_v4",
                }
            },
            "inputs": {
                "spatial_extent": {
                    "type": {
                        "type": "string[]",
                    }
                },
                "data_analysis_results": {
                    "type": "Directory"
                },
                "stageout_data_analysis_results": {
                    "type": "Directory"
                }
            },
            "outputs": {
                "split_tiles_results": {
                    "type": "Directory",
                    "outputBinding": {
                        "glob": "."
                    }
                },
                "tiles": {
                    "type": {
                        "type": "array",
                        "items": {
                            "type": "record",
                            "name": "TileRecord",
                            "fields": [
                                {
                                    "name": "spatial_extent",
                                    "type": "string[]"
                                },
                            ]
                        }
                    },
                    "outputBinding": {
                        "glob": "tiles/tiles.json",
                        "loadContents": True,
                        "outputEval": "${ return JSON.parse(self[0].contents); }"
                    }
                }
            }
        }

    def add_merge_results_graph(self):
        return {
            "class": "CommandLineTool",
            "id": "merge_results",
            "baseCommand": "python",
            "arguments": [
                "/app/merge_results.py",
                "--scatter_execution_results",
                "$(inputs.process_results)",
            ],
            "requirements": {
                "ResourceRequirement": {
                    "coresMax": 1,
                    "ramMax": 1024,
                }
            },
            "hints": {
                "DockerRequirement": {
                    "dockerPull": "pminel/zoo_reference_implementation_v4",
                }
            },
            "inputs": {
                "process_results": {
                    "type": "Directory[]"
                }
            },
            "outputs": {
                "execution_results": {
                    "type": "Directory",
                    "outputBinding": {
                        "glob": "."
                    }
                }
            }
        }

    def finalize_cwl(self, cwl):
        graphs = cwl["$graph"]
        for graph in graphs:
            if graph["class"] == "Workflow":
                print("Updating workflow graph")
                updated_workflow_graph = self.update_workflow_graph(graph)
                graphs.remove(graph)
                graphs.append(updated_workflow_graph)
            elif graph["class"] == "CommandLineTool" and graph["id"] == "process":
                print("Updating process graph")
                updated_process_graph = self.update_process_graph(graph)
                graphs.remove(graph)
                graphs.append(updated_process_graph)

        print("Adding stageout_data_analysis graph")
        graphs.append(self.add_stageout_data_analysis_graph())

        print("Adding split_tiles graph")
        graphs.append(self.add_split_tiles_graph())

        print("Adding merge_results graph")
        graphs.append(self.add_merge_results_graph())

        return cwl





def {{cookiecutter.workflow_id |replace("-", "_")  }}(conf, inputs, outputs):  # noqa

    try:
        logger.info(inputs)
        with open(
            os.path.join(
                pathlib.Path(os.path.realpath(__file__)).parent.absolute(),
                "app-package.cwl",
            ),
            "r",
        ) as stream:
            cwl = yaml.safe_load(stream)

        execution_handler = SimpleExecutionHandler(conf=conf)

        finalized_cwl = execution_handler.finalize_cwl(cwl)
        print("-- finalized_cwl --")
        print(finalized_cwl)
        print("-- finalized_cwl --")

        runner = ZooCalrissianRunner(
            cwl=finalized_cwl,
            conf=conf,
            inputs=inputs,
            outputs=outputs,
            execution_handler=execution_handler,
        )

        working_dir = os.path.join(conf["main"]["tmpPath"], runner.get_namespace_name())
        os.makedirs(
            working_dir,
            mode=0o777,
            exist_ok=True,
        )
        os.chdir(working_dir)

        exit_status = runner.execute()

        if exit_status == zoo.SERVICE_SUCCEEDED:
            """logger.info(f"Setting Collection into output key {list(outputs.keys())[0]}")
            outputs["stac_catalog"]["value"] = json.dumps(
                execution_handler.results, indent=2
            )"""
            return zoo.SERVICE_SUCCEEDED

        else:
            conf["lenv"]["message"] = zoo._("Execution failed")
            logger.error("Execution failed")
            return zoo.SERVICE_FAILED

    except Exception as e:

        logger.error("ERROR in processing execution template...")
        logger.error("Try to fetch the tool logs if any...")

        try:
            tool_logs = runner.execution.get_tool_logs()
            execution_handler.handle_outputs(None, None, None, tool_logs)
        except Exception as e:
            logger.error(f"Fetching tool logs failed! ({str(e)})")

        stack = traceback.format_exc()

        logger.error(stack)

        conf["lenv"]["message"] = zoo._(f"Exception during execution...\n{stack}\n")

        return zoo.SERVICE_FAILED

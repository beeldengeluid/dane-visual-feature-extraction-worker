import logging
import os
from pathlib import Path
import sys
from time import time
from base_util import validate_config
from dane import Document, Task, Result
from dane.base_classes import base_worker
from dane.config import cfg
from models import CallbackResponse, Provenance
from io_util import (
    transfer_output,
    obtain_input_file,
    delete_local_output,
    get_base_output_dir,
    get_source_id,
    get_download_dir,
    get_s3_base_url,
)
from pika.exceptions import ChannelClosedByBroker  # type: ignore
from feature_extraction import extract_features


"""
NOTE now the output dir created by by DANE (createDirs()) for the PATHS.OUT_FOLDER is not used:

- /mnt/dane-fs/output-files/03/d2/8a/03d28a03643a981284b403b91b95f6048576c234

Instead we put the output in:

- /mnt/dane-fs/output-files/visxp_prep/{source_id}
"""
logger = logging.getLogger()


class VisualFeatureExtractionWorker(base_worker):
    def __init__(self, config):
        logger.info(config)

        self.UNIT_TESTING = os.getenv("DW_VISXP_2_UNIT_TESTING", False)

        if not validate_config(config, not self.UNIT_TESTING):
            logger.error("Invalid config, quitting")
            sys.exit()

        # first make sure the config has everything we need
        # Note: base_config is loaded first by DANE,
        # so make sure you overwrite everything in your config.yml!
        try:
            self.DANE_DEPENDENCIES: list = (
                config.DANE_DEPENDENCIES if "DANE_DEPENDENCIES" in config else []
            )

            # read from default DANE settings
            self.DELETE_INPUT_ON_COMPLETION: bool = config.INPUT.DELETE_ON_COMPLETION
            self.DELETE_OUTPUT_ON_COMPLETION: bool = config.OUTPUT.DELETE_ON_COMPLETION
            self.TRANSFER_OUTPUT_ON_COMPLETION: bool = (
                config.OUTPUT.TRANSFER_ON_COMPLETION
            )

        except AttributeError:
            logger.exception("Missing configuration setting")
            sys.exit()

        # check if the file system is setup properly
        if not self.validate_data_dirs(
            get_download_dir(), get_base_output_dir()
        ):  # TODO: is this relevant for worker 2?
            logger.info("ERROR: data dirs not configured properly")
            if not self.UNIT_TESTING:
                sys.exit()

        # we specify a queue name because every worker of this type should
        # listen to the same queue
        self.__queue_name = "VISXP_EXTRACT"  # this is the queue that receives the work and NOT the reply queue
        # self.DANE_DOWNLOAD_TASK_KEY = "DOWNLOAD"
        self.__binding_key = (
            "#.VISXP_EXTRACT"  # ['Video.VISXP_PREP', 'Sound.VISXP_PREP']
        )
        self.__depends_on = self.DANE_DEPENDENCIES  # TODO make this part of DANE lib?

        if not self.UNIT_TESTING:
            logger.warning("Need to initialize the VISXP_EXTRACT service")

        super().__init__(
            self.__queue_name,
            self.__binding_key,
            config,
            self.__depends_on,
            auto_connect=not self.UNIT_TESTING,
            no_api=self.UNIT_TESTING,
        )

    """----------------------------------INIT VALIDATION FUNCTIONS ---------------------------------"""

    def validate_data_dirs(
        self, input_dir: str, visxp_output_dir: str
    ) -> bool:  # TODO: add model dir
        i_dir = Path(input_dir)
        o_dir = Path(visxp_output_dir)

        if not os.path.exists(i_dir.parent.absolute()):
            logger.info(
                f"{i_dir.parent.absolute()} does not exist. Make sure BASE_MOUNT_DIR exists before retrying"
            )
            return False

        # make sure the input and output dirs are there
        try:
            os.makedirs(i_dir, 0o755)
            logger.info("created VisXP input dir: {}".format(i_dir))
        except FileExistsError as e:
            logger.info(e)

        try:
            os.makedirs(o_dir, 0o755)
            logger.info("created VisXP output dir: {}".format(o_dir))
        except FileExistsError as e:
            logger.info(e)

        return True

    """----------------------------------INTERACTION WITH DANE SERVER ---------------------------------"""

    # DANE callback function, called whenever there is a job for this worker
    def callback(self, task: Task, doc: Document) -> CallbackResponse:
        logger.info("Receiving a task from the DANE server!")
        logger.info(task)
        logger.info(doc)

        # step 0: Create provenance object
        provenance = Provenance(
            activity_name="dane-visual-feature-extraction-worker",
            activity_description="Apply VisXP feature extraction to keyframes + spectograms",
            start_time_unix=time(),
            processing_time_ms=-1,
            input_data={},
            output_data={},
        )

        # obtain the input file
        # TODO make sure to download the output from S3
        output_file_path, download_provenance = obtain_input_file(self.handler, doc)
        if not output_file_path:
            return {
                "state": 500,
                "message": "Could not download the input from S3",
            }
        if download_provenance and provenance.steps:
            provenance.steps.append(download_provenance)

        input_file_path = output_file_path
        output_path = "TODO"  # TODO think of this

        # step 1: apply model to extract features
        proc_result = extract_features(
            input_file_path,
            model_path=cfg.VISXP_EXTRACT.MODEL_PATH,
            model_config_file=cfg.VISXP_EXTRACT.MODEL_CONFIG_PATH,
            output_path=output_path,
        )

        # step 2: raise exception on failure
        if proc_result.state != 200:
            logger.error(f"Could not process the input properly: {proc_result.message}")
            # something went wrong inside the VisXP work processor, return that response here
            return {"state": proc_result.state, "message": proc_result.message}

        if proc_result.provenance:
            if not provenance.steps:
                provenance.steps = []
            provenance.steps.append(proc_result.provenance)

        # step 3: process returned successfully, generate the output
        input_file = "*"
        source_id = get_source_id(
            input_file
        )  # TODO: this worker does not necessarily work per source, so consider how to capture output group

        # step 4: transfer the output to S3 (if configured so)
        transfer_success = True
        if self.TRANSFER_OUTPUT_ON_COMPLETION:
            transfer_success = transfer_output(source_id)

        if (
            not transfer_success
        ):  # failure of transfer, impedes the workflow, so return error
            return {
                "state": 500,
                "message": "Failed to transfer output to S3",
            }

        # step 5: clear the output files (if configured so)
        delete_success = True
        if self.DELETE_OUTPUT_ON_COMPLETION:
            delete_success = delete_local_output(source_id)

        if (
            not delete_success
        ):  # NOTE: just a warning for now, but one to keep an EYE out for
            logger.warning(f"Could not delete output files: {output_path}")

        # step 6: save the results back to the DANE index
        self.save_to_dane_index(
            doc,
            task,
            get_s3_base_url(source_id),
            provenance=provenance,
        )
        return {
            "state": 200,
            "message": "Successfully generated VisXP data for the next worker",
        }

    # TODO adapt to VisXP
    def save_to_dane_index(
        self,
        doc: Document,
        task: Task,
        s3_location: str,
        provenance: Provenance,
    ) -> None:
        logger.info("saving results to DANE, task id={0}".format(task._id))
        # TODO figure out the multiple lines per transcript (refresh my memory)
        r = Result(
            self.generator,
            payload={
                "doc_id": doc._id,
                "task_id": task._id if task else None,
                "doc_target_id": doc.target["id"],
                "doc_target_url": doc.target["url"],
                "s3_location": s3_location,
                "provenance": provenance.to_json(),
            },
            api=self.handler,
        )
        r.save(task._id)


# Start the worker
# passing --run-test-file will run the whole process on the files in cfg.VISXP_EXTRACT.TEST_FILES
if __name__ == "__main__":
    from argparse import ArgumentParser
    from base_util import LOG_FORMAT

    # first read the CLI arguments
    parser = ArgumentParser(description="dane-visual-feature-extraction-worker")
    parser.add_argument(
        "--run-test-file", action="store", dest="run_test_file", default="n", nargs="?"
    )
    parser.add_argument("--log", action="store", dest="loglevel", default="INFO")
    args = parser.parse_args()

    # initialises the root logger
    logging.basicConfig(
        stream=sys.stdout,  # configure a stream handler only for now (single handler)
        format=LOG_FORMAT,
    )

    # setting the loglevel
    log_level = args.loglevel.upper()
    logger.setLevel(log_level)
    logger.info(f"Logger initialized (log level: {log_level})")
    logger.info(f"Got the following CMD line arguments: {args}")

    # see if the test file must be run
    if args.run_test_file != "n":
        logger.info("Running feature extraction with VISXP_EXTRACT.TEST_INPUT_PATH ")
        if cfg.VISXP_EXTRACT and cfg.VISXP_EXTRACT.TEST_INPUT_PATH:
            visxp_fe = extract_features(
                input_path=cfg.VISXP_EXTRACT.TEST_INPUT_PATH,
                model_path=cfg.VISXP_EXTRACT.MODEL_PATH,
                model_config_file=cfg.VISXP_EXTRACT.MODEL_CONFIG_PATH,
                output_path=cfg.FILESYSTEM.OUTPUT_DIR,
            )
            if visxp_fe.provenance:
                logger.info(
                    "Successfully processed example files "
                    f"in {visxp_fe.provenance.processing_time_ms}ms"
                )
            else:
                logger.info(f"Error: {visxp_fe.state}: {visxp_fe.message}")
        else:
            logger.error("Please configure an input file in VISXP_PREP.TEST_INPUT_FILE")
            sys.exit()
    else:
        logger.info("Starting the worker")
        # start the worker
        w = VisualFeatureExtractionWorker(cfg)
        try:
            w.run()
        except ChannelClosedByBroker:
            """
            (406, 'PRECONDITION_FAILED - delivery acknowledgement on channel 1 timed out.
            Timeout value used: 1800000 ms.
            This timeout value can be configured, see consumers doc guide to learn more')
            """
            logger.critical(
                "Please increase the consumer_timeout in your RabbitMQ server"
            )
            w.stop()
        except (KeyboardInterrupt, SystemExit):
            w.stop()

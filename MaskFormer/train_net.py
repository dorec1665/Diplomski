# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MaskFormer Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import argparse
import glob
import pickle
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
import itertools
import logging
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Set

import numpy as np
import onnx
import torch
import onnx_tensorrt.backend as backend
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger
import tensorrt as trt

# MaskFormer
from mask_former import (
    DETRPanopticDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_mask_former_config,
)



class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
        ]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
        # DETR-style dataset mapper for COCO panoptic segmentation
        elif cfg.INPUT.DATASET_MAPPER_NAME == "detr_panoptic":
            mapper = DETRPanopticDatasetMapper(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former")
    return cfg

def benchmark_cuda(model, images, nwarmup=50):
    print("Warm up ...")
    with torch.no_grad():
        for i in range(nwarmup):
            features = model(images[i])
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    st = time.time()
    with torch.no_grad():
        for i in range(1, len(images)):
            start_time = time.time()
            pred_loc = model(images[i])
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i % 10 == 0:
                print('Iteration %d/%d, avg batch time %.2f ms' % (i, len(images), np.mean(timings) * 1000))

    print("--- %s seconds ---" % (np.sum(timings)))
    print("Input shape:", images[0].shape)
    print('Average throughput: %.2f images/second' % (len(images) / (np.mean(timings)*len(images))))

def benchmark(model, images, nwarmup=50):
    print("Warm up ...")
    with torch.no_grad():
        for i in range(nwarmup):
            features = model.run(images[i])
    torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for i in range(1, len(images)):
            start_time = time.time()
            pred_loc = model.run(images[i])
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i % 10 == 0:
                print('Iteration %d/%d, avg batch time %.2f ms' % (i, len(images), np.mean(timings) * 1000))

    print("Total time: %s seconds" % (np.sum(timings)))
    print("Input shape:", images[0].shape)
    print('Average throughput: %.2f images/second' % (len(images) / (np.mean(timings)*len(images))))

def prepare_image(image, scale_factor):
    image_np = np.array(image, dtype=np.float32) / 255.0
    image_tensor = torch.tensor(image_np)
    image_tensor = image_tensor.permute(2, 0, 1)
    image_tensor = torch.unsqueeze(image_tensor, dim=0)
    image_tensor = F.interpolate(image_tensor, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    image_tensor = torch.squeeze(image_tensor)
    image_tensor = image_tensor * 255
    image_tensor = torch.tensor(image_tensor, dtype=torch.int32)
    return image_tensor.to("cuda")

def prepare_image_np(image, scale_factor):
    image_np = np.array(image, dtype=np.float32) / 255.0
    image_tensor = torch.tensor(image_np)
    image_tensor = image_tensor.permute(2, 0, 1)
    image_tensor = torch.unsqueeze(image_tensor, dim=0)
    image_tensor = F.interpolate(image_tensor, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    image_tensor = torch.squeeze(image_tensor)
    image_tensor = image_tensor * 255
    image_tensor = torch.tensor(image_tensor, dtype=torch.int32)
    return np.array(image_tensor, dtype=np.int32)


def start_benchmark(args, model, image, trt_model_path, tensor_db, np_db):
    colormap = create_cityscapes_colormap()
    if args.cuda:
        print("CUDA BENCHMARK:")
        output = model(image)
        output = torch.argmax(output[0]["sem_seg"], dim=0).cpu().numpy()
        output = colorize_prediction(output, colormap)
        plt.imsave(args.res_path, output)
        benchmark_cuda(model, images=tensor_db)
    if args.compile:
        x = image
        torch.out = model(x)
        torch.onnx.export(model, x, trt_model_path, export_params=True, opset_version=16,
                          do_constant_folding=False, input_names=["input"], output_names=["output"])
    else:
        print("TENSORRT BENCHMARK:")

        if args.build_engine_type == 1:
            print("Building engine 1")
            model = onnx.load(trt_model_path)
            engine = backend.prepare(model, device='CUDA:' + str(torch.cuda.current_device()), max_workspace_size=1<<32)

            if args.trt_res_path:
                output = engine.run(np.array(image.cpu()))
                output = torch.tensor(output[0])
                output = torch.argmax(output, dim=0).cpu().numpy()
                output = colorize_prediction(output, colormap)
                plt.imsave(args.trt_res_path, output)
            benchmark(engine, images=np_db)
        elif args.build_engine_type == 2:
            print("Building engine 2")

            builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 32  # 4GB
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

            # Parse the ONNX model to populate the TensorRT network
            parser = trt.OnnxParser(network, builder.logger)
            with open(trt_model_path, 'rb') as model:
                parser.parse(model.read())

            # Build the engine
            engine = builder.build_engine(network, config)
            #print("ENGINE BUILT")
            context = engine.create_execution_context()

            input_data = np.array(image.cpu())

            # Allocate GPU memory
            d_input = cuda.mem_alloc(1 * input_data.nbytes)
            output_shape = (19, 512, 1024)  # Shape of the output
            output_size = int(np.prod(output_shape) * np.dtype(np.float32).itemsize)  # Size in bytes
            d_output = cuda.mem_alloc(output_size)

            stream = cuda.Stream()

            start = time.time()
            for i in range(1,len(np_db)):
                input_data = np.array(image.cpu())
                cuda.memcpy_htod_async(d_input, input_data, stream)
                context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
                #output_data = np.empty(output_shape, dtype=np.float32)  # Output data array
                #cuda.memcpy_dtoh_async(output_data, d_output, stream)
                #stream.synchronize()
            print("END: " + str(time.time() - start))
        print("Benchmark finished!")
        return

def prepare_dataset(path, device, scale_factor):
    dataset_as_tensor = []
    dataset_as_np = []
    print("Preparing dataset...")
    for filename in glob.iglob(path + '**/*.png', recursive=True):
        image = Image.open(filename)
        image_tensor = prepare_image(image, scale_factor)
        image_np = prepare_image_np(image, scale_factor)
        dataset_as_tensor.append(image_tensor)
        dataset_as_np.append(image_np)
    print("Dataset prepared!")
    return dataset_as_tensor, dataset_as_np

def create_cityscapes_colormap():
    # Initialize the colormap with zeros
    colormap = np.zeros((256, 3), dtype=int)

    # Mapping class labels to their corresponding colors
    colormap[0] = [128, 64, 128]    # Road
    colormap[1] = [244, 35, 232]    # Sidewalk
    colormap[2] = [70, 70, 70]      # Building
    colormap[3] = [102, 102, 156]   # Wall
    colormap[4] = [190, 153, 153]   # Fence
    colormap[5] = [153, 153, 153]   # Pole
    colormap[6] = [250, 170, 30]    # Traffic Light
    colormap[7] = [220, 220, 0]     # Traffic Sign
    colormap[8] = [107, 142, 35]    # Vegetation
    colormap[9] = [152, 251, 152]   # Terrain
    colormap[10] = [70, 130, 180]   # Sky
    colormap[11] = [220, 20, 60]    # Person
    colormap[12] = [255, 0, 0]      # Rider
    colormap[13] = [0, 0, 142]      # Car
    colormap[14] = [0, 0, 70]       # Truck
    colormap[15] = [0, 60, 100]     # Bus
    colormap[16] = [0, 80, 100]     # Train
    colormap[17] = [0, 0, 230]      # Motorcycle
    colormap[18] = [119, 11, 32]    # Bicycle
    # Other classes can be added if necessary

    return colormap


def colorize_prediction(prediction, colormap):
    colorized = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)

    for label in range(0, len(colormap)):
        mask = prediction == label
        colorized[mask] = colormap[label]

    return colorized


def main(args):
    cfg = setup(args)
    trt_model_path = args.trt_path
    device = args.device

    if args.eval_only:
        model = Trainer.build_model(cfg)
        state = pickle.load(open('model_final_38c00c.pkl', 'rb'))
        for k, v in state["model"].items():
            v = torch.tensor(v)
            state["model"][k] = v
        model.load_state_dict(state["model"])

        model.eval()
        model.to(device)
        shape = (3, 128, 256)
        image = Image.open("lindau_37.png")
        image_tensor = prepare_image(image, args.sample_factor)
        dataset_as_tensor, dataset_as_np = prepare_dataset("/mnt/sda/datasets/cityscapes/cityscapes/leftImg8bit/test/bielefeld/",
                                                           "cpu", args.sample_factor)
        #dataset_as_tensor = None
        #dataset_as_np = None
        print("Number of devices: " + str(torch.cuda.device_count()))
        print("Number of current device: " + str(torch.cuda.current_device()))
        print("Using device: " + torch.cuda.get_device_name(torch.cuda.current_device()))

        start_benchmark(args, model, image_tensor, trt_model_path, dataset_as_tensor, dataset_as_np)

        print("KRAJ")
        # DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        #     cfg.MODEL.WEIGHTS, resume=args.resume
        # )
        #res = Trainer.test(cfg, model)
        # if cfg.TEST.AUG.ENABLED:
        #     res.update(Trainer.test_with_TTA(cfg, model))
        # if comm.is_main_process():
        #     verify_results(cfg, res)
        # return res

    # trainer = Trainer(cfg)
    # trainer.resume_or_load(resume=args.resume)
    # return trainer.train()


if __name__ == "__main__":
    torch.cuda.set_device(0)
    parser = default_argument_parser()
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--device", type=str)
    parser.add_argument("--sample-factor", type=float)
    parser.add_argument("--trt-path", type=str)
    parser.add_argument("--trt-res-path", type=str)
    parser.add_argument("--res-path", type=str)
    parser.add_argument("--build-engine-type", type=int)
    args = parser.parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


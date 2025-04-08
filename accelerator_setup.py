import logging
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs

def setup_accelerator_and_logging(args):
    """Initializes Accelerator, logging, WandB, device, and model dtype."""
    logger = logging.getLogger(__name__) # Logger for this module

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
        log_with="wandb" if not args.disable_wandb else None
    )

    # Setup logging level based on process rank
    # Reconfigure root logger based on rank using force=True
    log_level = logging.INFO if accelerator.is_main_process else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )
    logger.info(f"Logging level set to {log_level}")

    if accelerator.is_main_process:
        logger.info(f"Accelerator state: {accelerator.state}")
        logger.info(f"Using {accelerator.num_processes} processes.")
        logger.info(f"Effective batch size: {args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")

    # Initialize WandB tracker via Accelerator
    if accelerator.is_main_process and not args.disable_wandb:
         wandb_kwargs = {}
         if args.wandb_run_name:
             wandb_kwargs["name"] = args.wandb_run_name
         try:
            accelerator.init_trackers(
                project_name=args.wandb_project,
                config=vars(args),
                init_kwargs={"wandb": wandb_kwargs}
            )
            logger.info(f"Initialized WandB tracker for project: {args.wandb_project}")
         except Exception as e:
             logger.error(f"Failed to initialize WandB tracker via Accelerator: {e}. Disabling WandB.")
             accelerator.log_with = None # Ensure accelerator knows wandb is disabled

    device = accelerator.device
    logger.info(f"Process {accelerator.process_index} using device: {device}")

    # Determine Model dtype
    model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    logger.info(f"Using model dtype: {model_dtype}")

    return accelerator, device, model_dtype 
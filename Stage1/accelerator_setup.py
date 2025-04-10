import logging
from accelerate import Accelerator, DistributedDataParallelKwargs

# Setup logger for this module (optional, if setup function logs directly)
logger = logging.getLogger(__name__)

def setup_accelerator_and_logging(args):
    """Initializes Accelerator, sets up logging, and initializes trackers."""

    # Initialize Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
        log_with="wandb" if not args.disable_wandb else None
    )

    # Setup logging level based on process rank
    log_level = logging.INFO if accelerator.is_main_process else logging.WARNING
    # Configure root logger
    logging.basicConfig(
        level=log_level, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True # Force re-configuration if already configured
    )
    logger.info(f"Logging level set to {log_level}") # Use the module logger

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
            # Make sure project name is passed correctly
            accelerator.init_trackers(
                project_name=args.wandb_project,
                config=vars(args),
                init_kwargs={"wandb": wandb_kwargs}
            )
            logger.info(f"Initialized WandB tracker for project: {args.wandb_project}")
         except Exception as e:
             logger.error(f"Failed to initialize WandB tracker via Accelerator: {e}. Disabling WandB.")
             # Ensure wandb logging is skipped later if init fails by setting accelerator's log_with correctly
             accelerator.log_with = None # Turn off logging if init failed

    # Log device info after setup
    logger.info(f"Process {accelerator.process_index} using device: {accelerator.device}")

    return accelerator 